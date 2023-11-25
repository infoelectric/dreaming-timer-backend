import { spawn } from "child_process";
import path from "path";

const scriptPath: string = path.join(
  __dirname,
  "../",
  "../",
  "scripts",
  "ImagePreprocessing.py"
);

// 이미지 처리를 위한 파이썬 스크립트 실행 함수 (프로미스 사용)
const detectionScript = async (imagePath: string): Promise<number> => {
  return new Promise<number>((resolve, reject) => {
    const process = spawn("python3", [scriptPath, imagePath]);

    let stdoutData = "";

    process.stdout.on("data", (data) => {
      stdoutData += data.toString();
    });

    process.stderr.on("data", (data) => {
      console.error(`Python script stderr: ${data.toString()}`);
    });

    process.on("close", (code) => {
      if (code !== 0) {
        console.error(`Python script exited with code ${code}`);
        reject(new Error("Python script execution failed"));
      } else {
        const averageGrayscale = parseFloat(stdoutData);
        resolve(averageGrayscale);
      }
    });

    process.on("error", (error) => {
      console.error(`Error executing python script: ${error.message}`);
      reject(error);
    });
  });
};

export default detectionScript;
