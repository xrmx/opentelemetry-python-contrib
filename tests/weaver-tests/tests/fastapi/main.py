# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}
