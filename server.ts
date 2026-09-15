import dotenv from "dotenv";
import path from "path";
dotenv.config();
dotenv.config({ path: path.resolve(process.cwd(), `.env.${process.env.NODE_ENV || 'development'}`), override: true });

import express, { Request, Response } from "express";
import { createServer as createViteServer } from "vite";
import axios from "axios";
import fs from "fs";

const app = express();
const PORT = 3350;

let scans: any[] = [];
const DATA_DIR = path.join(process.cwd(), "data");
const SCANS_FILE = path.join(DATA_DIR, "scans.json");

if (!fs.existsSync(DATA_DIR)) {
  fs.mkdirSync(DATA_DIR, { recursive: true });
}

if (fs.existsSync(SCANS_FILE)) {
  try {
    scans = JSON.parse(fs.readFileSync(SCANS_FILE, "utf-8"));
  } catch (e) {
    console.error("Error reading scans file:", e);
  }
}

const saveScans = () => {
  try {
    fs.writeFileSync(SCANS_FILE, JSON.stringify(scans, null, 2));
  } catch (e) {
    console.error("Error saving scans file:", e);
  }
};

app.use(express.json({ limit: "50mb" }));

app.get("/api/scans", (req: Request, res: Response) => {
  res.json([...scans].sort((a, b) => b.timestamp - a.timestamp));
});

app.post("/api/scans", (req: Request, res: Response) => {
  const newScan = {
    ...req.body,
    id: Math.random().toString(36).substr(2, 9),
    timestamp: Date.now(),
    sent: false,
  };
  scans.push(newScan);
  saveScans();
  res.json(newScan);
});

app.delete("/api/scans/:id", (req: Request, res: Response) => {
  scans = scans.filter((s) => s.id !== req.params.id);
  saveScans();
  res.json({ success: true });
});

app.post("/api/scans/:id/send", async (req: Request, res: Response) => {
  const scan = scans.find((s) => s.id === req.params.id);
  if (!scan) {
    return res.status(404).json({ error: "Scan not found" });
  }

  const webhookUrl = process.env.WEBHOOK_URL;
  if (!webhookUrl) {
    return res.status(500).json({ error: "WEBHOOK_URL is not configured on the server" });
  }

  try {
    await axios.post(webhookUrl, {
      image: scan.image,
      timestamp: scan.timestamp,
      scanId: scan.id,
    });

    scan.sent = true;
    saveScans();
    res.json({ success: true });
  } catch (error) {
    console.error("Webhook send error:", error);
    res.status(502).json({ error: "Failed to reach webhook" });
  }
});

async function startServer() {
  if (!process.env.WEBHOOK_URL) {
    console.warn("WARNING: WEBHOOK_URL is not set. Sending scans will fail until it's configured in .env.");
  }

  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
