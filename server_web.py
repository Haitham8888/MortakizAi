import os
import uvicorn

# تشغيل الواجهة (الويب)
os.environ["RUN_MODE"] = "web"

import server

if __name__ == "__main__":
    print("✅ Web server ready")
    uvicorn.run(server.app, host=server.config.SERVER_HOST, port=server.config.SERVER_PORT, log_level=server.config.LOG_LEVEL)
