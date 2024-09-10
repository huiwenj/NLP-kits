import axios from "axios";

const service = axios.create({
  baseURL: "http://localhost:8000/api/v1",
});

service.interceptors.response.use((response) => {
  if (response.status === 200) {
    return response.data;
  } else {
    return Promise.reject(response);
  }
});

export default service;
