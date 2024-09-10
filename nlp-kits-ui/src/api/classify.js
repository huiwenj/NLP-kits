import http from "@/util/http";

export const train = () => {
  return http.get("/classify/train");
};

export const predict = (word) => {
  return http.get("/classify/predict", { params: { word } });
};
