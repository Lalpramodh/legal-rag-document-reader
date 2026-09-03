import io
import os
import re
import json
import hashlib
import traceback
from typing import List

import faiss
import fitz
import numpy as np
import pytesseract
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from fastembed import TextEmbedding
from PIL import Image
from pydantic import BaseModel, EmailStr
from groq import AsyncGroq

load_dotenv()

app = FastAPI(
    title="Legal RAG Document Reader",
    version="1.0.0"
)

# ---------------------------- Configuration ----------------------------

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "legal_assistant")

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

if not MONGO_URI:
    # Local development fallback.
    # Production must provide MONGO_URI.
    MONGO_URI = "mongodb://localhost:27017"

client = AsyncIOMotorClient(
    MONGO_URI,
    serverSelectionTimeoutMS=10000
)

db = client[MONGO_DB]

documents_collection = db.documents
users_collection = db.users
queries_collection = db.queries


# ---------------------------- Models ----------------------------

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.1-8b-instant"
)

# FastEmbed uses ONNX instead of PyTorch/SentenceTransformers.
# This is much lighter for Render's free 512 MB instance.
embedding_model = TextEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

groq_client = (
    AsyncGroq(api_key=GROQ_API_KEY)
    if GROQ_API_KEY
    else None
)

# all-MiniLM-L6-v2 = 384 dimensions
EMBEDDING_DIM = 384


# ---------------------------- FAISS Stores ----------------------------

# 1. Bundled legal knowledge base
# 2. User-uploaded documents

kb_index = faiss.IndexFlatL2(EMBEDDING_DIM)
uploaded_index = faiss.IndexFlatL2(EMBEDDING_DIM)

kb_store = {}
uploaded_store = {}


# ---------------------------- Paths ----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KB_JSON = os.path.join(
    BASE_DIR,
    "cleaned_legal_data.json"
)

KB_EMBEDDINGS = os.path.join(
    BASE_DIR,
    "legal_embeddings.npy"
)


# ---------------------------- Utility Functions ----------------------------

def clean_text(text: str) -> str:
    """
    Normalize OCR/PDF text.
    """
    text = text.replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()


def create_embedding(text: str) -> np.ndarray:
    """
    Generate a single 384-dimensional embedding using FastEmbed.
    """

    embeddings = list(
        embedding_model.embed([text])
    )

    vector = np.asarray(
        embeddings[0],
        dtype="float32"
    )

    return vector.reshape(1, -1)


# ---------------------------- Knowledge Base ----------------------------

def load_knowledge_base() -> None:
    """
    Load the bundled legal Q&A knowledge base
    and its precomputed vectors.
    """

    if not (
        os.path.exists(KB_JSON)
        and os.path.exists(KB_EMBEDDINGS)
    ):
        print(
            "Knowledge base files not found; "
            "starting with an empty KB."
        )
        return

    try:
        with open(
            KB_JSON,
            "r",
            encoding="utf-8"
        ) as f:
            records = json.load(f)

        vectors = np.load(
            KB_EMBEDDINGS
        ).astype("float32")

        if len(records) != len(vectors):
            raise ValueError(
                "Knowledge base JSON and embedding counts "
                "do not match"
            )

        if vectors.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dimension mismatch. "
                f"Expected {EMBEDDING_DIM}, "
                f"got {vectors.shape[1]}"
            )

        kb_index.add(vectors)

        for i, record in enumerate(records):

            question = record.get(
                "question",
                ""
            )

            answer = record.get(
                "answer",
                ""
            )

            kb_store[i] = (
                f"Question: {question}\n"
                f"Answer: {answer}"
            )

        print(
            f"Loaded {len(records)} "
            "legal knowledge-base records."
        )

    except Exception as exc:

        print(
            f"Failed to load knowledge base: {exc}"
        )


# ---------------------------- Startup ----------------------------

@app.on_event("startup")
async def startup_event():
    """
    Load knowledge base and persistent
    uploaded-document vectors.
    """

    load_knowledge_base()

    try:

        docs = await documents_collection.find(
            {
                "embedding": {
                    "$exists": True
                }
            },
            {
                "filename": 1,
                "text": 1,
                "embedding": 1
            },
        ).to_list(length=None)

        for doc in docs:

            vector = np.asarray(
                doc["embedding"],
                dtype="float32"
            ).reshape(1, -1)

            if vector.shape[1] != EMBEDDING_DIM:
                continue

            idx = uploaded_index.ntotal

            uploaded_index.add(vector)

            uploaded_store[idx] = {
                "filename": doc.get(
                    "filename",
                    "document"
                ),
                "text": doc.get(
                    "text",
                    ""
                ),
            }

        print(
            f"Restored "
            f"{uploaded_index.ntotal} "
            "uploaded-document vectors."
        )

    except Exception as exc:

        print(
            f"Could not restore "
            f"uploaded documents: {exc}"
        )


# ---------------------------- Request Models ----------------------------

class ChatRequest(BaseModel):
    question: str


class LegalDocumentRequest(BaseModel):
    template: str
    clauses: str


class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


# ---------------------------- OCR / PDF ----------------------------

def extract_text_from_pdf_bytes(
    data: bytes
) -> str:

    """
    Extract text from a PDF.

    If the PDF is scanned/image-only,
    OCR is performed.
    """

    try:

        doc = fitz.open(
            stream=data,
            filetype="pdf"
        )

        text_parts = [
            page.get_text("text")
            for page in doc
        ]

        text = clean_text(
            " ".join(text_parts)
        )

        # OCR scanned PDFs
        if len(text) < 30:

            ocr_parts = []

            for page in doc:

                pix = page.get_pixmap(
                    matrix=fitz.Matrix(
                        1.5,
                        1.5
                    ),
                    alpha=False
                )

                image = Image.open(
                    io.BytesIO(
                        pix.tobytes("png")
                    )
                )

                ocr_parts.append(
                    pytesseract.image_to_string(
                        image
                    )
                )

            text = clean_text(
                " ".join(ocr_parts)
            )

        doc.close()

        return text

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Error extracting text "
                f"from PDF: {exc}"
            )
        )


def extract_text_from_image_bytes(
    data: bytes
) -> str:

    try:

        image = Image.open(
            io.BytesIO(data)
        )

        return clean_text(
            pytesseract.image_to_string(
                image
            )
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Error extracting text "
                f"from image: {exc}"
            )
        )


# ---------------------------- RAG ----------------------------

def add_uploaded_document(
    text: str,
    filename: str
) -> None:

    vector = create_embedding(text)

    idx = uploaded_index.ntotal

    uploaded_index.add(vector)

    uploaded_store[idx] = {
        "filename": filename,
        "text": text
    }


def search_index(
    index,
    store,
    query: str,
    top_k: int = 3
):

    if index.ntotal == 0:
        return []

    vector = create_embedding(query)

    distances, indices = index.search(
        vector,
        min(
            top_k,
            index.ntotal
        )
    )

    results = []

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        if idx < 0 or idx not in store:
            continue

        similarity = (
            1.0 /
            (
                1.0 +
                float(distance)
            )
        )

        results.append(
            (
                similarity,
                store[idx]
            )
        )

    return results


def retrieve_context(
    query: str,
    top_k: int = 3
):

    """
    Prefer uploaded-document context
    when it is more relevant than the KB.
    """

    uploaded = search_index(
        uploaded_index,
        uploaded_store,
        query,
        top_k
    )

    kb = search_index(
        kb_index,
        kb_store,
        query,
        top_k
    )

    best_uploaded = (
        uploaded[0][0]
        if uploaded
        else -1
    )

    best_kb = (
        kb[0][0]
        if kb
        else -1
    )

    if (
        uploaded
        and best_uploaded >=
        max(0.25, best_kb)
    ):

        return (
            "\n\n".join(
                item["text"]
                for score, item
                in uploaded
            ),
            "uploaded_document"
        )

    if kb:

        return (
            "\n\n".join(
                item
                for score, item
                in kb
            ),
            "knowledge_base"
        )

    return "", "none"


# ---------------------------- Groq LLM ----------------------------

async def generate_groq(
    prompt: str,
    max_output_tokens: int = 220
) -> str:

    if not groq_client:

        raise HTTPException(
            status_code=500,
            detail=(
                "GROQ_API_KEY "
                "is not configured."
            )
        )

    try:

        completion = (
            await groq_client
            .chat
            .completions
            .create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a careful "
                            "legal document assistant. "
                            "Answer clearly and accurately. "
                            "When context is provided, "
                            "use only that context "
                            "and do not invent facts."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.1,
                max_completion_tokens=max_output_tokens,
            )
        )

        return (
            completion
            .choices[0]
            .message
            .content
            .strip()
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Groq generation failed: "
                f"{exc}"
            )
        )


# ---------------------------- Summarization ----------------------------

async def summarize_text(
    text: str
) -> str:

    chunks = [
        text[i:i + 1800]
        for i in range(
            0,
            len(text),
            1800
        )
    ]

    summaries = []

    for chunk in chunks[:12]:

        summary = await generate_groq(
            (
                "Summarize the following "
                "legal document text clearly "
                "and concisely:\n\n"
                f"{chunk}"
            ),
            max_output_tokens=160,
        )

        summaries.append(summary)

    return " ".join(summaries)


# ---------------------------- Legal Document Generation ----------------------------

DOCUMENT_TEMPLATES = {

    "rental agreement": """
RENTAL AGREEMENT

This Rental Agreement is made and entered into on
[Effective Date] by and between:

Landlord: [Landlord Name], residing at
[Landlord Address].

Tenant: [Tenant Name], residing at
[Tenant Address].

1. RENTAL PROPERTY

The Landlord agrees to rent to the Tenant
the property located at [Rental Address]
("Premises").

2. ADDITIONAL CLAUSES

{clauses}
""",

    "land registration": """
LAND REGISTRATION AGREEMENT

This Land Registration Agreement is made and
entered into on [Effective Date] between:

Seller: [Seller Name], residing at
[Seller Address].

Buyer: [Buyer Name], residing at
[Buyer Address].

1. LAND DESCRIPTION

Land Address: [Land Address]

Total Area: [Land Size] acres

2. ADDITIONAL CLAUSES

{clauses}
""",

    "building registration": """
BUILDING REGISTRATION AGREEMENT

This Building Registration Agreement is made
on [Effective Date] between:

Owner: [Owner Name], residing at
[Owner Address].

Registrar: [Registrar Name], an official
representative of [City/Municipality].

1. PROPERTY DETAILS

Address: [Building Address]

Type: [Residential/Commercial]

2. ADDITIONAL CLAUSES

{clauses}
""",

    "lease agreement": """
LEASE AGREEMENT

This Lease Agreement is made and entered into
as of [Effective Date] by and between:

Landlord: [Landlord Name], residing at
[Landlord Address].

Tenant: [Tenant Name], residing at
[Tenant Address].

1. PREMISES

Address: [Property Address]

2. ADDITIONAL CLAUSES

{clauses}
"""
}


# ---------------------------- API ----------------------------

@app.get("/")
async def root():

    return {
        "message":
        "Legal RAG API is running"
    }


@app.get("/health")
async def health():

    mongo_ok = True

    try:

        await client.admin.command(
            "ping"
        )

    except Exception:

        mongo_ok = False

    return {

        "status": "ok",

        "mongodb": mongo_ok,

        "knowledge_base_vectors":
            int(kb_index.ntotal),

        "uploaded_vectors":
            int(uploaded_index.ntotal),
    }


# ---------------------------- Authentication ----------------------------

@app.post("/auth/signup")
async def signup(
    request: SignupRequest
):

    existing = await users_collection.find_one(
        {
            "email":
            request.email.lower()
        }
    )

    if existing:

        raise HTTPException(
            status_code=409,
            detail="Email already registered"
        )

    password_hash = hashlib.sha256(
        request.password
        .encode("utf-8")
    ).hexdigest()

    await users_collection.insert_one(
        {
            "name":
                request.name.strip(),

            "email":
                request.email.lower(),

            "password_hash":
                password_hash
        }
    )

    return {
        "success": True,
        "message":
            "Account created successfully"
    }


@app.post("/auth/login")
async def login(
    request: LoginRequest
):

    password_hash = hashlib.sha256(
        request.password
        .encode("utf-8")
    ).hexdigest()

    user = await users_collection.find_one(
        {
            "email":
                request.email.lower(),

            "password_hash":
                password_hash
        }
    )

    if not user:

        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    return {

        "success": True,

        "message":
            "Login successful",

        "user": {

            "name":
                user.get("name"),

            "email":
                user.get("email")
        }
    }


# ---------------------------- Legal Document Generator ----------------------------

@app.post("/generate-legal-doc/")
async def generate_legal_doc(
    request: LegalDocumentRequest
):

    template = request.template.lower()

    if template not in DOCUMENT_TEMPLATES:

        raise HTTPException(
            status_code=400,
            detail="Invalid template selected."
        )

    result = (
        DOCUMENT_TEMPLATES[template]
        .format(
            clauses=request.clauses
        )
    )

    return jsonable_encoder(
        {
            "legal_document":
                result
        }
    )


# ---------------------------- Document Upload ----------------------------

@app.post("/upload-legal-doc/")
async def upload_legal_doc(
    file: UploadFile = File(...)
):

    try:

        filename = (
            file.filename
            or "document"
        )

        print(
            f"[UPLOAD] Starting upload: {filename}"
        )

        extension = os.path.splitext(
            filename.lower()
        )[1]

        if extension not in {
            ".pdf",
            ".png",
            ".jpg",
            ".jpeg"
        }:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported file format. "
                    "Only PDF, JPG, and PNG "
                    "are allowed."
                )
            )

        data = await file.read()

        print(
            f"[UPLOAD] File read successfully: "
            f"{len(data)} bytes"
        )

        if extension == ".pdf":

            print("[UPLOAD] Extracting text from PDF...")

            extracted_text = (
                extract_text_from_pdf_bytes(
                    data
                )
            )

        else:

            print("[UPLOAD] Extracting text from image...")

            extracted_text = (
                extract_text_from_image_bytes(
                    data
                )
            )

        print(
            f"[UPLOAD] Text extraction complete: "
            f"{len(extracted_text)} characters"
        )

        if not extracted_text:

            raise HTTPException(
                status_code=400,
                detail=(
                    "No text could be extracted "
                    "from the document."
                )
            )

        print("[UPLOAD] Creating FastEmbed vector...")

        # Generate 384-dimensional embedding
        vector = create_embedding(
            extracted_text
        )

        print(
            f"[UPLOAD] Embedding created: "
            f"shape={vector.shape}, "
            f"dtype={vector.dtype}"
        )

        print("[UPLOAD] Saving document to MongoDB...")

        # Store in MongoDB
        await documents_collection.insert_one(
            {
                "filename":
                    filename,

                "text":
                    extracted_text,

                "embedding":
                    vector[0].tolist()
            }
        )

        print("[UPLOAD] MongoDB insert successful.")

        print("[UPLOAD] Adding document to FAISS index...")

        # Add to in-memory FAISS index
        add_uploaded_document(
            extracted_text,
            filename
        )

        print("[UPLOAD] FAISS indexing successful.")

        print("[UPLOAD] Generating summary with Groq...")

        # Generate summary using Groq
        summary = await summarize_text(
            extracted_text
        )

        print("[UPLOAD] Groq summary generated successfully.")

        return {

            "message":
                "Document processed and indexed successfully!",

            "filename":
                filename,

            "extracted_text":
                extracted_text[:500],

            "summary":
                summary
        }

    except HTTPException as exc:
    print(
        f"[UPLOAD HTTP ERROR] "
        f"status={exc.status_code}, "
        f"detail={exc.detail}"
    )
    raise

    except Exception as exc:

        print(
            "[UPLOAD ERROR] "
            f"{type(exc).__name__}: {exc}"
        )

        print(
            traceback.format_exc()
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Document upload failed: "
                f"{type(exc).__name__}: {exc}"
            )
        )


# ---------------------------- Chatbot ----------------------------

@app.post("/chatbot/")
async def answer_legal_question(
    request: ChatRequest
):

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    context, source = retrieve_context(
        question
    )

    if context:

        prompt = (
            "Answer the legal question "
            "using ONLY the provided context. "
            "If the context does not contain "
            "enough information, say that the "
            "document does not provide enough "
            "information. "
            "Do not invent facts.\n\n"
            f"Context:\n{context[:10000]}\n\n"
            f"Question: {question}"
        )

    else:

        prompt = (
            f"Answer briefly and clearly: "
            f"{question}"
        )

    answer = await generate_groq(
        prompt
    )

    await queries_collection.insert_one(
        {
            "question":
                question,

            "source":
                source,

            "answer":
                answer
        }
    )

    return {

        "answer":
            answer,

        "source":
            source
    }


# ---------------------------- CORS ----------------------------

if ALLOWED_ORIGINS == ["*"]:

    app.add_middleware(
        CORSMiddleware,

        allow_origins=["*"],

        allow_credentials=False,

        allow_methods=["*"],

        allow_headers=["*"],
    )

else:

    app.add_middleware(
        CORSMiddleware,

        allow_origins=
            ALLOWED_ORIGINS,

        allow_credentials=True,

        allow_methods=["*"],

        allow_headers=["*"],
    )


# ---------------------------- Shutdown ----------------------------

@app.on_event("shutdown")
async def shutdown_event():

    client.close()
