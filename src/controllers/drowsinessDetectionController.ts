import { Request, Response } from "express";

import { upload } from "@middleware/drowsinessDetectionMulter";

interface ImageData {
  imageName: string;
  imageUrl: string;
}

// 이미지 처리 라우트
export const postDetection = (req: Request, res: Response) => {
  // 업로드된 파일의 정보 출력
  console.log("업로드된 파일 정보:");
  console.log(req?.file);

  res.send({ message: "success" });
};
