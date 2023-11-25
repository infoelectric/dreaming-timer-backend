import express from "express";

import { postDetection } from "@controllers/drowsinessDetectionController";
import { upload } from "@middleware/drowsinessDetectionMulter";

const router = express.Router();

router.post("/", upload, postDetection);

export default router;
