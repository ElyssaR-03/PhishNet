import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const analyzeURL = async (url, model = 'random_forest') => {
  try {
    const response = await api.post('/analyze/url', { url, model });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to analyze URL');
  }
};

export const analyzeEmail = async (content, sender = '', model = 'random_forest') => {
  try {
    const response = await api.post('/analyze/email', { content, sender, model });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to analyze email');
  }
};

export const trainModels = async (n_samples = 1000) => {
  try {
    const response = await api.post('/train', { n_samples });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to train models');
  }
};

export const getModelsInfo = async () => {
  try {
    const response = await api.get('/models/info');
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to get models info');
  }
};

export const getEducationTips = async () => {
  try {
    const response = await api.get('/education/tips');
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.detail || 'Failed to get education tips');
  }
};

export const checkHealth = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    throw new Error('API is not available');
  }
};

export default api;
