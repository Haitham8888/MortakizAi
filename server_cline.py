import os
import uvicorn

# تشغيل Cline فقط (بدون الواجهة)
os.environ["RUN_MODE"] = "cline"

import server

if __name__ == "__main__":
    print("✅ Cline server ready")
    uvicorn.run(server.app, host=server.config.SERVER_HOST, port=8081, log_level=server.config.LOG_LEVEL)
