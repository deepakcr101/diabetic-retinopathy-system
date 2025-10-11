import axios from 'axios'

const api = axios.create({
  baseURL: '/', // Vite proxy will forward /api to Django
  headers: {
    'Content-Type': 'application/json',
  },
})

// Attach access token
api.interceptors.request.use(config => {
  const access = localStorage.getItem('access')
  if(access){
    config.headers = config.headers || {}
    config.headers.Authorization = `Bearer ${access}`
  }
  return config
})

// Response interceptor to try refresh on 401
let isRefreshing = false
let failedQueue = []

const processQueue = (error, token = null) => {
  failedQueue.forEach(p => {
    if(error) p.reject(error)
    else p.resolve(token)
  })
  failedQueue = []
}

api.interceptors.response.use(
  response => response,
  async error => {
    const originalRequest = error.config
    if(error.response && error.response.status === 401 && !originalRequest._retry){
      if(isRefreshing){
        return new Promise(function(resolve, reject){
          failedQueue.push({ resolve, reject })
        }).then(token => {
          originalRequest.headers.Authorization = 'Bearer ' + token
          return api(originalRequest)
        }).catch(err => Promise.reject(err))
      }

      originalRequest._retry = true
      isRefreshing = true
      const refresh = localStorage.getItem('refresh')
      if(!refresh) return Promise.reject(error)
      try{
        const res = await axios.post('/api/token/refresh/', { refresh })
        const newAccess = res.data.access
        localStorage.setItem('access', newAccess)
        api.defaults.headers.common['Authorization'] = 'Bearer ' + newAccess
        processQueue(null, newAccess)
        return api(originalRequest)
      }catch(err){
        processQueue(err, null)
        return Promise.reject(err)
      }finally{
        isRefreshing = false
      }
    }
    return Promise.reject(error)
  }
)

export default api
