# Variáveis de Ambiente com `--env-file-if-exists`

> Você já viu `node --env-file .env` nos projetos do módulo 02 (por exemplo, no [`01-smart-model-router-gateway`](../../../modulo02-integracao-apis-llms/01-smart-model-router-gateway/README.md)). Este doc parte daí e foca no que muda aqui: o `tsx` no meio do caminho, a variante `-if-exists` e o bug real que isso causou no OpsPilot.

---

## O bug que motivou este doc

Com o `.env` preenchido, a primeira execução real da arena falhou:

```text
> npm run arena -- "quantos alertas criticos estão disparando" --strategies react

> ops-pilot@1.0.0 arena
> tsx src/arena.ts quantos alertas criticos estão disparando --strategies react

Arena failed: OPENROUTER_API_KEY and OPENROUTER_MODEL are required to run an LLM strategy.
```

O arquivo estava lá, as variáveis também. O problema é que **ninguém lia o arquivo**. O script era `tsx src/arena.ts`, e nem o `tsx` nem o Node carregam `.env` por conta própria. Quem faz isso "sozinho" são frameworks como Next.js e Vite. O Node puro só carrega quando você pede.

Esse bug estava previsto: era a armadilha 3 do [roteiro-de-implementacao.md](./roteiro-de-implementacao.md). A implementação seguiu o `tasks.md`, que preservou os scripts originais, e o problema só apareceu na primeira execução com LLM de verdade. Os testes não pegaram porque eles **de propósito** não usam credenciais.

---

## O que é

Uma variável de ambiente é um par `NOME=valor` que o sistema operacional entrega ao processo quando ele nasce. No Node, ela aparece em `process.env.NOME`. O `.env` é só um **arquivo de texto** com esses pares, uma convenção para não precisar exportar tudo no terminal a cada sessão.

O ponto central: `process.env` é preenchido **antes** do seu código rodar. Logo, alguém precisa ler o `.env` e jogar os valores ali. Existem três formas de fazer isso:

| Quem carrega | Como | Disponível desde |
| --- | --- | --- |
| O Node, por flag de linha de comando | `node --env-file=.env app.js` | Node 20.6 |
| O Node, por flag, sem erro se o arquivo faltar | `node --env-file-if-exists=.env app.js` | Node 22.9 |
| O seu código, em runtime | `process.loadEnvFile(".env")` | Node 20.12 / 21.7 |
| Uma biblioteca | `import "dotenv/config"` | — (**proibido** pela nossa constituição) |

Analogia: o `.env` é a lista de compras presa na geladeira. `process.env` é a sacola. A lista não pula sozinha para a sacola; alguém precisa pegá-la ao sair de casa. O `--env-file` é o Node pegando a lista **na porta**, antes de sair.

---

## Como está sendo usado neste projeto

### Os scripts (`package.json`)

```json
"dev": "node --env-file-if-exists=.env --import tsx src/index.ts",
"arena": "node --env-file-if-exists=.env --import tsx src/arena.ts",
"bench": "node --env-file-if-exists=.env --import tsx src/bench.ts",
"seed": "node --env-file-if-exists=.env --import tsx src/services/seed.ts",
"test": "node --import tsx --test \"src/**/*.test.ts\"",
```

> JSON não aceita comentários, por isso a explicação dos scripts fica aqui, e não no `package.json`.

Lendo o comando `arena` da esquerda para a direita:

```text
node                          ← quem executa é o Node de verdade
  --env-file-if-exists=.env   ← 1º: lê .env (se existir) e preenche process.env
  --import tsx                ← 2º: registra o tsx como "loader" → o Node passa a entender .ts
  src/arena.ts                ← 3º: só agora o seu código roda, com process.env já pronto
```

### Por que `node --import tsx` e não `tsx`?

Antes era `tsx src/arena.ts`, em que o `tsx` é o executável. Agora quem executa é o `node`, e o `tsx` entra só como **plugin** (`--import`) para transpilar TypeScript. A vantagem é que todas as flags nativas do Node ficam disponíveis do jeito documentado pelo Node: `--env-file-if-exists`, `--watch`, `--inspect`, `--test`. O script `test` já usava esse formato (`node --import tsx --test`). Agora todos os scripts seguem o mesmo padrão.

> Também funcionaria `tsx --env-file-if-exists=.env src/arena.ts`, porque o `tsx` repassa flags desconhecidas ao Node. Preferimos o `node --import tsx` por deixar explícito quem faz o quê.

### Quem consome (`src/agents/model.ts`)

```ts
export function createModel(): ChatOpenAI {
  const apiKey = process.env.OPENROUTER_API_KEY;
  const model = process.env.OPENROUTER_MODEL;
  // `!valor` (e não `valor === undefined`) também barra string vazia — o caso de copiar
  // o .env.example sem preencher, que faria o erro aparecer só no provedor.
  if (!apiKey || !model) {
    throw new AgentConfigurationError(
      "OPENROUTER_API_KEY and OPENROUTER_MODEL are required to run an LLM strategy.",
    );
  }
```

Repare que a fábrica **não sabe** que existe um `.env`. Ela só lê `process.env`. Isso é bom: em produção ou em CI, as variáveis podem vir de outro lugar (secrets do GitHub Actions, painel da Vercel, `docker run -e`) e o código não muda. O `if (!apiKey || !model)` falha cedo, com uma mensagem clara, em vez de deixar o erro estourar lá no OpenRouter. Isso também resolve a armadilha 4 do roteiro (string vazia).

### Por que o `test` **não** carrega o `.env`

Foi uma decisão de propósito. A constituição exige testes determinísticos e sem rede, e o [`model.test.ts`](../src/agents/model.test.ts) verifica justamente o erro **sem** credenciais:

```ts
test("model factory reports missing provider configuration without making a request", () => {
  const apiKey = process.env.OPENROUTER_API_KEY;
  const modelName = process.env.OPENROUTER_MODEL;
  delete process.env.OPENROUTER_API_KEY;
  delete process.env.OPENROUTER_MODEL;
```

Se o `test` carregasse o `.env`, o resultado da suíte passaria a depender da máquina de quem roda. Na sua máquina, teria a chave; no CI, não.

---

## `--env-file` × `--env-file-if-exists`

| Situação | `--env-file=.env` | `--env-file-if-exists=.env` |
| --- | --- | --- |
| `.env` existe | carrega | carrega |
| `.env` **não** existe | **erro** e o processo nem inicia (`.env: not found`) | segue em frente sem carregar nada (só avisa no terminal: `.env not found. Continuing without it.`) |

Por que o `-if-exists` é melhor aqui: num clone novo do repositório, ou no CI, não existe `.env`. Com `--env-file`, até um comando que **não precisa** de chave quebraria antes de começar. Com `-if-exists`, o processo inicia, e quem precisa da chave (a `createModel`) reclama com uma mensagem que diz exatamente o que falta.

### Regras do formato que valem saber

- **Comentários** com `#` são aceitos, como no nosso [`.env.example`](../.env.example).
- **Quem já está no ambiente vence.** Se o terminal já tem `OPENROUTER_MODEL` definido, o valor do arquivo **não** sobrescreve. Isso é útil para trocar de modelo pontualmente: `$env:OPENROUTER_MODEL = "outro/modelo"; npm run arena -- ...`.
- **Vários arquivos:** pode repetir a flag (`--env-file=.env --env-file-if-exists=.env.local`). Em caso de conflito entre os arquivos, o último vence.
- O arquivo é lido **relativo ao diretório atual** (`cwd`). Rodando `npm run` na raiz do projeto, é o `.env` da raiz.

---

## Outras situações em que isso se aplica

O mesmo padrão (Node carrega e o código só lê `process.env`) resolve vários casos que vamos encontrar no curso:

1. **Credenciais de banco.** O adapter Sequelize (`src/models/sequelize-store.ts`) vai precisar de host, usuário e senha do MySQL. O caminho é `DB_HOST`, `DB_USER` e `DB_PASSWORD` no `.env`, e o script que conecta usa `--env-file-if-exists`.
2. **Porta e configuração de servidor.** A unidade 03 traz uma API (`POST /chat`): `PORT=3000` no `.env` e `app.listen(Number(process.env.PORT ?? 3000))` no código.
3. **Modo desenvolvimento com watch.** As flags se combinam: `node --env-file-if-exists=.env --watch --import tsx src/index.ts`. Observação: o `.env` é lido só na partida. Se mudar o `.env`, reinicie.
4. **Ambientes em camadas.** Uma base comum mais um override pessoal fora do git: `node --env-file=.env --env-file-if-exists=.env.local app.js`. O primeiro é obrigatório; o segundo, opcional.
5. **Testes de integração separados.** Os testes unitários ficam sem `.env`, como o nosso `test`. Um script à parte, como `"test:integration": "node --env-file-if-exists=.env.test --import tsx --test \"src/**/*.int.test.ts\""`, usa um `.env.test` com chaves de um ambiente de teste.
6. **CI e Docker.** No GitHub Actions, as variáveis vêm de `secrets` via `env:`. No Docker, de `docker run -e` ou do `environment:` do compose. Nesses lugares não há `.env`, e o `-if-exists` permite que o **mesmo** script funcione sem alteração.
7. **Scripts avulsos** (seed, migração, backfill). Qualquer script que toque serviço externo segue o mesmo formato. O nosso `seed` já está assim.
8. **Quando não dá para usar flag.** Alguns ambientes controlam a linha de comando por você, como um runner de terceiros. Nesses casos, o plano B nativo é chamar `process.loadEnvFile()` na primeira linha do ponto de entrada, envolvido em `try/catch` caso o arquivo não exista.

---

## Como montar isso do zero num projeto novo

Passo a passo para um projeto TypeScript com Node 22.9 ou mais recente, sem `dotenv`.

**1. Dependências**

```bash
npm init -y
npm install -D typescript tsx @types/node
```

**2. Arquivos de ambiente**

`.env.example` (versionado; documenta **o que** é preciso, sem valores):

```bash
# Copie para .env e preencha
API_KEY=
PORT=3000
```

`.gitignore` (o `.env` real **nunca** vai para o git):

```gitignore
.env
.env.*
!.env.example
```

Depois: `cp .env.example .env` e preencha.

**3. Scripts** (`package.json`)

```json
"scripts": {
  "dev": "node --env-file-if-exists=.env --watch --import tsx src/index.ts",
  "start": "node --env-file-if-exists=.env --import tsx src/index.ts",
  "test": "node --import tsx --test \"src/**/*.test.ts\""
}
```

**4. Leitura centralizada e validada** (um único arquivo lê `process.env`, como a nossa `createModel`)

```ts
// src/config.ts
export function loadConfig() {
  const apiKey = process.env.API_KEY;
  if (!apiKey) throw new Error("API_KEY é obrigatória. Copie .env.example para .env e preencha.");
  return { apiKey, port: Number(process.env.PORT || 3000) };
}
```

Com Zod (que já usamos), a validação fica declarativa:

```ts
import { z } from "zod";

const Env = z.object({
  API_KEY: z.string().min(1),
  PORT: z.coerce.number().default(3000),
});
export const env = Env.parse(process.env);
```

> Cuidado com o `export const env = Env.parse(...)` no topo do módulo: ele valida **no import**, o que derruba testes que só importam o arquivo. Por isso a nossa `createModel` valida dentro da função, só quando alguém pede um modelo de verdade.

**5. Confirmar que carregou, sem imprimir o segredo**

```bash
node --env-file-if-exists=.env -e "console.log('API_KEY definida?', !!process.env.API_KEY)"
```

Nunca use `console.log(process.env)`: isso joga todas as chaves no terminal e, às vezes, em logs persistidos.

---

## Armadilhas

- **"Criei o `.env` e continua dando erro."** Confira se o comando executado tem `--env-file...`. A primeira linha que o `npm run` imprime (`> node --env-file-if-exists=.env ...`) mostra o comando real.
- **`tsx src/arena.ts` ou `npx tsx ...` direto** ignoram o `.env`. Use os scripts do `npm run`.
- **Node antigo.** Antes da 22.9, `--env-file-if-exists` dá `bad option`. Confira com `node --version`. A constituição fixa Node 22 LTS, e as versões 22.9 em diante servem.
- **`npm run` no Git Bash do Windows** pode falhar com `Cannot find module ...npm-cli.js`. É um problema de caminho entre o Git Bash e o gerenciador de versões do Node, não do projeto. Rode pelo PowerShell.
- **Variável "fantasma" do terminal.** Como o ambiente vence o arquivo, um `OPENROUTER_MODEL` antigo exportado na sessão ignora o que está no `.env`. Abra um terminal novo ou rode `Remove-Item Env:OPENROUTER_MODEL`.

---

## Referências no projeto

| Conceito | Arquivo | O que observar |
| --- | --- | --- |
| Scripts que carregam o `.env` | [`package.json`](../package.json) | `node --env-file-if-exists=.env --import tsx` em `dev`, `arena`, `bench`, `seed`; `test` sem `.env` |
| Único ponto que lê as credenciais | [`src/agents/model.ts`](../src/agents/model.ts) | `createModel()`: lê `process.env` e falha cedo com `AgentConfigurationError` |
| Teste sem credenciais | [`src/agents/model.test.ts`](../src/agents/model.test.ts) | Remove as variáveis e espera o erro, sem rede |
| Modelo do `.env` | [`.env.example`](../.env.example) | Variáveis exigidas, com comentários |
| `.env` fora do git | [`.gitignore`](../.gitignore) | `.env`, `.env.*`, `!.env.example` |
| Regra "sem `dotenv`" | [`.specify/memory/constitution.md`](../.specify/memory/constitution.md) | "carregue variáveis de ambiente usando o suporte nativo do Node.js" |
| Onde o bug foi previsto | [roteiro-de-implementacao.md](./roteiro-de-implementacao.md) | Armadilhas 3 e 4 |
