# AGENTS.md

## Repository identity

- Recommended public repository name: `job-agent-backend`
- Recommended product description: FastAPI-based AI backend for job search, CV matching, Bedrock tool-calling, and MLflow-traced agent evaluation.
- This repository is an agent backend, not a frontend and not a generic LangChain starter.

## Agent rules

1. Ground all explanations in the actual codebase.
2. Do not invent frameworks, services, or deployment steps not present in the repository.
3. Do not describe this project as using LangChain, LangGraph, or Bedrock AgentCore unless code is added that clearly does so.
4. Describe the runtime as a custom FastAPI + AWS Lambda + Mangum backend using Amazon Bedrock.
5. Treat all secrets, account identifiers, MLflow endpoints, and database URIs as private.

## Current implementation facts

- API runtime is FastAPI served through AWS Lambda using Mangum.
- The agent loop is custom and tool-driven.
- LLM inference is handled through Amazon Bedrock.
- The configured production model family is Claude 3.5 Haiku.
- The backend exposes:
  - chat and tool-calling flows
  - CV matching using embedding-based similarity
  - direct job search against Supabase-backed data
- MLflow is used for tracing, evaluation, prompt iteration, and experiment logging.

## Documentation expectations

- Keep README statements implementation-accurate.
- Separate core architecture from optional deployment details.
- Prefer clear explanations of purpose, request flow, tools, evaluation, and observability.
- Explicitly note when something is a legacy filename, example dataset, or optional appendix.

## Safe cleanup policy

Safe to remove when preparing for public release:

- local `.env` files
- generated eval results
- credential-bearing local scripts
- sample diagrams and scratch files
- internal one-off notes without long-term developer value

Use extra caution around:

- deployment scripts
- MLflow evaluation steps
- router behavior
- prompt policy and confidence logic
- environment variable names used by production code
