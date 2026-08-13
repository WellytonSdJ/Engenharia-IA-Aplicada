import { MultiServerMCPClient } from '@langchain/mcp-adapters';
import { createGoogleTrendsTool } from '../tools/googleTrendsTool.ts';
import { SerpAPIService } from './serpApiService.ts';
import { config } from '../config.ts';

export const getMCPTools = async () => {
  // Servidor MCP filesystem conectado por herança do padrão de 01-multiple-mcp-tools,
  // mas nenhum prompt deste projeto pede leitura/escrita de arquivos — fica disponível
  // para o agente sem ser usado pelo domínio (recomendação de conteúdo)
  const mcpClient = new MultiServerMCPClient({
    filesystem: {
      transport: 'stdio',
      command: 'npx',
      args: ['-y', '@modelcontextprotocol/server-filesystem', process.cwd()],
    },
  });

  const mcpTools = await mcpClient.getTools();

  const serpAPIService = new SerpAPIService(config.serpAPIConfig);
  const googleTrendsTool = createGoogleTrendsTool(serpAPIService);

  // Mistura tool MCP (filesystem) com tool nativa do LangChain (google_trends) na mesma lista
  return [...mcpTools, googleTrendsTool];
};
