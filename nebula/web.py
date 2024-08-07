from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
import asyncio
import uvicorn

app = FastAPI()

rootdir = Path(__file__).parent


@app.get("/static/{filename}")
async def serve_static(filename: str):
    return FileResponse(rootdir / "static" / filename)


@app.get("/")
async def index():
    return FileResponse(rootdir / "static" / "index.html")


@app.get("/frames.proto")
async def frames_proto():
    return FileResponse(rootdir / "frames.proto")


async def start_server():
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()
    return server


async def main():
    return asyncio.create_task(start_server())


if __name__ == "__main__":
    asyncio.run(main())
