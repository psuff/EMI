import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from routes import router

# Create FastAPI app
app = FastAPI()

# Include routes
app.include_router(router)

# Read HTML content from file
with open("EMI_cyberpunk.html", "r") as file:
    html_content = file.read()

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content=html_content, status_code=200)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
