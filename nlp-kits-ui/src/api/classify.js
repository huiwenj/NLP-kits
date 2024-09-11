import http from "@/util/http";

export const wordTrain = () => {
  return http.get("/classify/word/train");
};

export const wordPredict = (word) => {
  return http.get("/classify/word/predict", { params: { word } });
};

export const cityTrain = () => {
  return http.get("/classify/city/train");
};

export const cityPredict = (city) => {
  return http.get("/classify/city/predict", { params: { city } });
};

export const cityList = () => {
  return http.get("/classify/city/list");
};
