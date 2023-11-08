import express, { Request, Response } from "express";

import cors from "cors";
import dotenv from "dotenv";

dotenv.config();

const app = express();

app.use(express.urlencoded({ extended: true }));
app.use(express.json());
app.use(cors());

app.set("port", process.env.PORT || 3000);

app.listen(app.get("port"), () =>
  console.log(`서버 리스닝 포트: ${app.get("port")}`)
);
