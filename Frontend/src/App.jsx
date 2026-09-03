import React from "react";
import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import Home from "./pages/Home";
import Document from "./pages/Document";
import Chatbot from "./pages/Chatbot";
import Generate from "./pages/Generate";
function App() {
    return (
        <Routes>
            <Route path="/" element={<Layout />}>
                <Route index element={<Home />}/>
                <Route path="/upload-legal-doc" element={<Document/>}/>
                <Route path="/chatbot" element={<Chatbot/>}/>
                <Route path="/generate-legal-doc" element={<Generate />}/>
            </Route>
        </Routes>
    );
}

export default App;
