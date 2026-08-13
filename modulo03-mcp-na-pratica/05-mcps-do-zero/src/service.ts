
import { randomBytes, createCipheriv, createDecipheriv, scryptSync } from 'node:crypto';

// salt fixo (não por-mensagem): simplificação didática — garante que a mesma passphrase
// sempre derive a mesma chave, sem precisar armazenar/transmitir um salt junto do ciphertext
const SALT = 'mcp-encrypter-salt';

// scrypt em vez de hash direto da passphrase: AES-256 exige chave de exatamente 32 bytes,
// e scrypt é deliberadamente caro de computar — dificulta força bruta contra a passphrase
function deriveKey(passphrase: string): Buffer {
    return scryptSync(passphrase, SALT, 32);
}

export function encrypt(text: string, key: string): string {
    // IV novo a cada chamada: em CBC, reusar IV com a mesma chave vaza padrões entre mensagens
    // (blocos iniciais iguais viram ciphertext igual) — por isso a mesma mensagem cifrada duas
    // vezes produz saídas diferentes
    const iv = randomBytes(16);
    const cipher = createCipheriv('aes-256-cbc', deriveKey(key), iv);
    const encrypted = Buffer.concat([
        cipher.update(Buffer.from(text, 'utf8')),
        cipher.final(),
    ]);
    // IV não é secreto, só precisa ser único — vai embutido na própria saída para o decrypt reaproveitar
    return `${iv.toString('hex')}:${encrypted.toString('hex')}`;
}

export function decrypt(encryptedText: string, key: string): string {
    // rest.join(':') em vez de assumir só 2 segmentos: defensivo contra um eventual ':' extra
    // no ciphertext (na prática não ocorre, pois hex só usa 0-9a-f)
    const [ivHex, ...rest] = encryptedText.split(':');
    const iv = Buffer.from(ivHex, 'hex');
    const encrypted = Buffer.from(rest.join(':'), 'hex');
    const decipher = createDecipheriv('aes-256-cbc', deriveKey(key), iv);
    // decipher.final() lança exceção se o padding não bater — cobre tanto passphrase errada
    // quanto ciphertext malformado; quem trata isso é o handler da tool em mcp.ts, não aqui
    const decrypted = Buffer.concat([
        decipher.update(encrypted),
        decipher.final(),
    ]);
    return decrypted.toString('utf8');
}
