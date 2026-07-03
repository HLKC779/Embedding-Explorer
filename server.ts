import express from 'express';
import path from 'path';
import { createServer as createViteServer } from 'vite';
import { GoogleGenAI } from '@google/genai';

async function startServer() {
  const app = express();
  const PORT = 3000;

  // Initialize Gemini client on the server
  const ai = new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      }
    }
  });

  app.use(express.json());

  // API endpoints FIRST
  app.post('/api/interpret', async (req, res) => {
    try {
      const { token, x, y, z, w } = req.body;
      const prompt = `Interpret the semantic meaning of a token named "${token}" in a high-dimensional embedding space.
Coordinates:
X (Valence): ${Number(x).toFixed(2)} (-10 is negative, 10 is positive)
Y (Intensity): ${Number(y || 0).toFixed(2)} (-10 is calm, 10 is intense)
Z (Formality): ${Number(z || 0).toFixed(2)} (-10 is informal, 10 is formal)
W (Complexity): ${Number(w || 0).toFixed(2)} (-10 is simple, 10 is complex)

Provide a concise (1-2 sentence) interpretation of what this vector represents in natural language.`;

      const response = await ai.models.generateContent({
        model: 'gemini-3.5-flash',
        contents: prompt,
      });

      res.json({ text: response.text || "No interpretation available." });
    } catch (error: any) {
      console.error("Interpret Error:", error);
      res.status(500).json({ error: error.message || "Failed to generate interpretation" });
    }
  });

  app.post('/api/embed', async (req, res) => {
    try {
      const { query } = req.body;
      const result = await ai.models.embedContent({
        model: 'gemini-embedding-2-preview',
        contents: [{ parts: [{ text: query }] }],
      });
      
      const queryVector = result.embeddings[0].values;
      res.json({ values: queryVector });
    } catch (error: any) {
      console.error("Embed Error:", error);
      res.status(500).json({ error: error.message || "Failed to generate embedding" });
    }
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error("Failed to start server", err);
});
