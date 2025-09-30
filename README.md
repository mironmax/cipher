# Cipher Custom - Personal Local Setup

<div align="center">

<img src="./assets/cipher-logo.png" alt="Cipher Agent Logo" width="400" />

<p align="center">
<em>Docker-optimized Cipher setup for personal use with Claude Code and other MCP clients</em>
</p>

</div>

## What is This?

This is a streamlined, **fully local** Docker-based version of [Byterover Cipher](https://github.com/campfirein/cipher) - an open-source memory layer for AI coding agents.

**Key difference**: Everything runs locally on your machine - all databases, storage, and services are included and containerized. Only LLM API calls go to external services (Groq/OpenAI for inference, Gemini for embeddings).

Optimized for:
- **Personal local development** (not for production/team use)
- **Complete local infrastructure** - no external database services needed
- **MCP integration** with Claude Code, Cursor, Windsurf, etc.
- **Ready to use** - just add API keys and go

## Quick Start

### Prerequisites

- Docker & Docker Compose
- 2 API keys: Gemini (embeddings) + Groq/OpenAI (LLM)

### Setup (5 minutes)

1. **Clone and configure**
```bash
git clone <your-repo-url>
cd cipher-custom

# Copy and edit environment file
cp .env.example .env
# Add your API keys to .env
```

2. **Get your API keys** (free options available)
   - **Gemini**: https://aistudio.google.com/app/apikey (free tier)
   - **Groq**: https://console.groq.com/keys (generous free tier, RECOMMENDED)
   - OR **OpenAI**: https://platform.openai.com/api-keys (paid)

3. **Configure `.env`**
```bash
# Required: Add these to .env
GEMINI_API_KEY=your-gemini-api-key-here
OPENAI_API_KEY=your-groq-api-key-here  # Or OpenAI key
OPENAI_BASE_URL=https://api.groq.com/openai/v1  # Or https://api.openai.com/v1
```

4. **Start services**
```bash
docker-compose up -d
```

5. **Verify it's running**
```bash
docker-compose ps
# All services should show "Up"
```

## What's Included (All Local!)

This setup includes all necessary databases and services running in Docker containers:

- **cipher-app**: Main Cipher service with MCP server
- **cipher-qdrant**: Vector database for embeddings (local)
- **cipher-neo4j**: Knowledge graph database (local)
- **cipher-postgres**: Session storage (local)

**No external database services required!** All data stays on your machine in the `./data` directory.

## Using with Claude Code

Add to your Claude Code MCP configuration:

**macOS**: `~/Library/Application\ Support/Claude/claude_desktop_config.json`
**Linux**: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "cipher": {
      "command": "docker",
      "args": ["exec", "-i", "cipher-app", "node", "/app/build/index.js", "--mode", "mcp"],
      "env": {
        "MCP_SERVER_MODE": "default"
      }
    }
  }
}
```

Restart Claude Code, and you'll see `ask_cipher` tool available.

## Configuration

The setup uses these tested defaults:

- **LLM**: Groq's `openai/gpt-oss-120b` (OpenAI's open model, fast & cheap)
- **Embeddings**: Gemini `embedding-001` (3072 dimensions)
- **Vector Store**: Qdrant (local container)
- **Knowledge Graph**: Neo4j (local container)
- **Session Storage**: PostgreSQL (local container)

All databases run in Docker with generic passwords (safe for local use - services not exposed outside Docker network).

## Logs & Data

Access logs and data:
```bash
# View logs
docker-compose logs -f cipher-app

# Log files
tail -f data/logs/cipher-mcp.log

# Data persistence (all local!)
./data/           # All persistent data
./data/logs/      # Application logs
./data/events/    # Event persistence
./data/qdrant/    # Vector database
./data/neo4j/     # Knowledge graph
./data/postgres/  # Session storage
```

## Maintenance

```bash
# Stop services
docker-compose down

# Restart services
docker-compose restart

# Rebuild after code changes
docker-compose up --build -d

# Clean everything (caution: deletes all local data!)
docker-compose down -v
rm -rf ./data
```

## Troubleshooting

**"Connection refused" errors**:
- Check services are running: `docker-compose ps`
- Check logs: `docker-compose logs cipher-app`

**"Permission denied" on files**:
- Set UID/GID in `.env` to match your user: `id -u` and `id -g`

**MCP tool not showing in Claude Code**:
- Restart Claude Code after adding configuration
- Check cipher-app is running: `docker-compose ps cipher-app`

## Architecture

```
Claude Code (MCP Client)
    ↓
cipher-app (MCP Server + Agent)
    ↓
├── cipher-qdrant (vectors) ← LOCAL
├── cipher-neo4j (knowledge graph) ← LOCAL
└── cipher-postgres (sessions) ← LOCAL
```

**Everything except LLM API calls runs on your machine.**

## What's Different from Upstream?

This fork focuses on fully local, personal use:
- ✅ **All databases included** (Qdrant, Neo4j, PostgreSQL)
- ✅ **No external services** except LLM APIs
- ✅ Docker-first setup (no npm/pnpm needed)
- ✅ Personal use (removed team features)
- ✅ Opinionated config (Groq + Gemini)
- ✅ Simplified setup (3-5 minute start)

Upstream Cipher offers more flexibility, team features, and cloud deployment options.

## Credits

Built on [Byterover Cipher](https://github.com/campfirein/cipher) by the [Byterover team](https://byterover.dev/).

## License

Elastic License 2.0. See [LICENSE](LICENSE) for full terms.