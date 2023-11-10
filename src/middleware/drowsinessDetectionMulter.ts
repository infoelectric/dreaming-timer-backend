import multer from "multer";
import path from "path";

// 이미지를 저장할 경로와 파일명 설정
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, path.join(__dirname, "../", "../", "public", "face"));
  },
  filename: (req, file, cb) => {
    const timestamp = Date.now(); // 현재 시간 정보
    cb(null, `face-${timestamp}.png`); // 시간 정보를 파일명에 포함
  },
});

// multer 미들웨어 생성
export const upload = multer({ storage }).fields([
  { name: "image", maxCount: 1 },
]);
