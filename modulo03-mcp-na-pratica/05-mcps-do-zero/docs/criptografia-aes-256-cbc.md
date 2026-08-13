# As decisões de criptografia por trás das tools

Toda a lógica de criptografia vive isolada em `src/service.ts`, sem nenhuma dependência do SDK de MCP — só o módulo nativo `node:crypto`. Isso é proposital: a camada de protocolo (`mcp.ts`) não precisa saber como AES funciona, e a lógica de criptografia não precisa saber o que é uma tool.

```typescript
// src/service.ts
import { randomBytes, createCipheriv, createDecipheriv, scryptSync } from 'node:crypto';

const SALT = 'mcp-encrypter-salt';

function deriveKey(passphrase: string): Buffer {
    return scryptSync(passphrase, SALT, 32);
}
```

## Por que derivar a chave com `scrypt` em vez de usar a passphrase direto

AES-256 exige uma chave de exatamente 32 bytes — uma passphrase digitada por um humano ("my-secret-key") não tem esse tamanho nem essa distribuição de bytes. `scryptSync(passphrase, SALT, 32)` resolve os dois problemas de uma vez: sempre produz 32 bytes, e é uma função de derivação de chave (KDF) desenhada para ser cara de computar — dificulta ataques de força bruta contra a passphrase, ao contrário de simplesmente fazer um hash rápido (ex: SHA-256) da string.

**O salt é fixo (`'mcp-encrypter-salt'`), não aleatório por chamada** — isso é uma simplificação didática, não uma prática recomendada para um sistema de produção. Um salt fixo e hardcoded no código elimina a proteção que o salt normalmente oferece contra rainbow tables entre instalações diferentes deste servidor. Ele existe aqui só para que a mesma passphrase sempre derive a mesma chave, sem precisar armazenar ou transmitir o salt junto com a mensagem cifrada.

## Por que um IV novo a cada chamada de `encrypt`

```typescript
export function encrypt(text: string, key: string): string {
    const iv = randomBytes(16);   // 16 bytes = tamanho de bloco do AES — gerado de novo em toda chamada
    const cipher = createCipheriv('aes-256-cbc', deriveKey(key), iv);
    ...
    return `${iv.toString('hex')}:${encrypted.toString('hex')}`;
}
```

Em modo CBC, reutilizar o mesmo IV com a mesma chave para textos diferentes vaza padrões: os primeiros blocos idênticos produzem ciphertext idêntico, dando a um observador pistas sobre o conteúdo. Gerar um IV aleatório novo em cada chamada garante que a mesma mensagem, cifrada duas vezes com a mesma passphrase, produza saídas completamente diferentes — por isso o README do projeto destaca esse comportamento como característica, não como bug.

Como o IV não é secreto (só precisa ser único), ele é simplesmente concatenado ao ciphertext no formato de saída `<IV em hex>:<ciphertext em hex>`, separados por `:`. Isso evita ter que gerenciar o IV como um parâmetro extra — a própria string de saída carrega tudo que `decrypt` precisa, além da passphrase.

## Por que `decrypt` faz `split(':')` com `rest.join(':')`

```typescript
export function decrypt(encryptedText: string, key: string): string {
    const [ivHex, ...rest] = encryptedText.split(':');
    const iv = Buffer.from(ivHex, 'hex');
    const encrypted = Buffer.from(rest.join(':'), 'hex');
    ...
}
```

O ciphertext em hex nunca contém `:` (hex só usa `0-9a-f`), então na prática um `split(':')` simples já bastaria. O código usa `[ivHex, ...rest]` e depois `rest.join(':')` como uma forma defensiva de garantir que, se por algum motivo a string de entrada tiver mais de um `:`, só o primeiro segmento é tratado como IV e o resto é remontado como ciphertext — em vez de quebrar silenciosamente pegando só o segundo pedaço de um `split` de tamanho 2.

## Por que os erros de `decrypt` viram `isError: true` na tool, não uma exceção não tratada

`decipher.final()` lança exceção quando o padding do texto decifrado não bate — o que acontece tanto para uma passphrase errada (a chave derivada não corresponde ao IV/ciphertext) quanto para um ciphertext malformado. `service.ts` deixa essa exceção propagar; é a tool `decrypt_message`, em `mcp.ts`, que captura o erro e o transforma numa resposta MCP válida com `isError: true` — ver [construindo-mcp-server-do-zero.md](./construindo-mcp-server-do-zero.md) para o porquê dessa escolha no nível do protocolo.

---

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Derivação de chave | [src/service.ts](../src/service.ts) | `deriveKey` — `scryptSync(passphrase, SALT, 32)` |
| IV aleatório por chamada | [src/service.ts](../src/service.ts) | `randomBytes(16)` dentro de `encrypt` (não no escopo do módulo) |
| Formato de saída | [src/service.ts](../src/service.ts) | `${iv.toString('hex')}:${encrypted.toString('hex')}` |
| Descrição do algoritmo exposta ao cliente | [src/mcp.ts](../src/mcp.ts) | Resource `encryption://info` |
| Caminho de erro (não coberto por teste automatizado hoje) | [src/mcp.ts](../src/mcp.ts) | `catch` na tool `decrypt_message` — dispara quando `decipher.final()` falha (passphrase errada ou ciphertext malformado) |
