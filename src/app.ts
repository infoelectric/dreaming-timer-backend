import express from "express";

import cors from "cors";
import dotenv from "dotenv";

import DrowsinessDetectionRouter from "@routers/drowsinessDetectionRouter";

dotenv.config();

const app = express();

app.use(express.urlencoded({ extended: true }));
app.use(express.json());
app.use(cors());

app.set("port", process.env.PORT || 3000);

app.use("/api/drowsinessdetection", DrowsinessDetectionRouter);

app.listen(app.get("port"), () =>
  console.log(`서버 리스닝 포트: ${app.get("port")}`)
);
