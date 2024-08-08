#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import aiohttp
import asyncio
import os
import sys

from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantContextAggregator,
    LLMUserContextAggregator,
)
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAITTSService, OpenAILLMContext, OpenAILLMService
from pipecat.transports.network.websocket_server import (
    WebsocketServerParams,
    WebsocketServerTransport,
)
from pipecat.vad.silero import SileroVADAnalyzer
from openai.types.chat import ChatCompletionToolParam
from loguru import logger
from dotenv import load_dotenv

from nebula.memory import SQLiteVecMemory

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")
db = SQLiteVecMemory("memory.db")


async def user_append_memory(llm, args):
    db.write_memory(args["note"])
    return [{"role": "system", "content": "Note added, say done when finished."}]


async def user_query_memory(llm, args):
    query = args["query"]
    query = db.query_memory(query)

    notes = []
    for row in query:
        notes.append(f"- {row['created_at']}: {row['note']}")
    notes = "\n".join(notes)

    return [
        {
            "role": "system",
            "content": f"""
    Notes found that match the query:
    {notes}
    
    You can answer to the user now.
    """,
        }
    ]


async def main():
    async with aiohttp.ClientSession() as session:
        transport = WebsocketServerTransport(
            params=WebsocketServerParams(
                audio_out_sample_rate=24000,
                audio_out_enabled=True,
                add_wav_header=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_audio_passthrough=True,
            )
        )

        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
        )

        llm.register_function(
            "write_memory",
            user_append_memory,
        )
        llm.register_function(
            "query_memory",
            user_query_memory,
        )
        tools = [
            ChatCompletionToolParam(
                type="function",
                function={
                    "name": "write_memory",
                    "description": "Write a note to memory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "note": {
                                "type": "string",
                                "description": "Note to append",
                            },
                        },
                    },
                    "required": ["note"],
                },
            ),
            ChatCompletionToolParam(
                type="function",
                function={
                    "name": "query_memory",
                    "description": "Search notes in memory using vector similarity",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "What you want to search. It will be used to search the notes in memory, using vector similarity.",
                            },
                        },
                    },
                },
            ),
        ]

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

        tts = OpenAITTSService(
            aiohttp_session=session,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant in a WebRTC call. Your objective is to showcase your abilities clearly and effectively. Since your output will be converted to audio, avoid using special characters. Focus on capturing and recalling key points, and respond to the user's statements in a concise, creative, and helpful manner",
            },
        ]

        context = OpenAILLMContext(messages, tools)

        tma_in = LLMUserContextAggregator(context)
        tma_out = LLMAssistantContextAggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),  # Websocket input from client
                stt,  # Speech-To-Text
                tma_in,  # User responses
                llm,  # LLM
                tts,  # Text-To-Speech
                transport.output(),  # Websocket output to client
                tma_out,  # LLM responses
            ]
        )

        task = PipelineTask(pipeline)

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            messages.append({"role": "system", "content": "Say connected, then wait."})
            await task.queue_frames([LLMMessagesFrame(messages)])

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
