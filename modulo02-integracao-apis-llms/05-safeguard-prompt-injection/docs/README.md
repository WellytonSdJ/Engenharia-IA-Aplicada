# Documentação — Safeguard & Prompt Injection

Documentação de estudo do projeto `03-safeguard-prompt-injection`.

**Chegando agora? Comece por [00-START-HERE.md](./00-START-HERE.md).**

---

## Índice

| Documento | O que cobre |
| --- | --- |
| [00-START-HERE.md](./00-START-HERE.md) | Trilha de leitura ordenada, mapa do código, fluxo do projeto |
| [prompt-injection.md](./prompt-injection.md) | O que é prompt injection, vetores de ataque, demonstração prática — com e sem guardrails |
| [guardrails.md](./guardrails.md) | O que são guardrails, como funcionam, por que são defesa determinística e não probabilística |
| [mcp.md](./mcp.md) | Model Context Protocol: o que é, como conecta ferramentas ao LLM, por que amplia a superfície de ataque |
| [rbac.md](./rbac.md) | Role-Based Access Control: sistema admin/member, permissões, por que RBAC via prompt falha sem guardrail |
| [langgraph.md](./langgraph.md) | StateGraph deste projeto: guardrails_check → chat/blocked, estado de segurança, roteamento condicional |
| [glossario.md](./glossario.md) | Todos os termos do projeto com definições — referência rápida |

---

## O que este projeto demonstra

Este projeto existe para provar, com código executável, uma coisa:

> **System prompts não são controle de acesso. Código é.**

Ele roda o mesmo system prompt em dois modos — com e sem guardrails — e mostra o que acontece quando um usuário tenta contornar as restrições via prompt injection.

- **Sem guardrails:** o LLM pode ser manipulado a executar ações não autorizadas, mesmo com regras explícitas no prompt
- **Com guardrails:** um modelo safeguard analisa o input *antes* de chegar ao executor — e bloqueia a requisição sem nem chamar o LLM principal

## Contexto do projeto

Chatbot com sistema de controle de acesso com:

- **LangGraph** orquestrando o fluxo de segurança (`guardrails_check → chat/blocked`)
- **OpenRouter** como gateway de LLMs (modelo executor + modelo safeguard)
- **MCP (Model Context Protocol)** expondo ferramentas de sistema de arquivos ao LLM
- **RBAC** separando permissões entre admin e member
- **Prompt injection** demonstrada em dois modos: vulnerável (`--unsafe`) e protegido (padrão)
