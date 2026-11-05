# Teacher Charles — Análise inicial de viabilidade

*Novembro de 2026*

## 1. Visão

Um chatbot para praticar e estudar inglês. A persona é um professor de inglês nativo e experiente, que também fala português muito bem. O objetivo é que todos os recursos (modelo de linguagem, reconhecimento de fala, síntese de voz) rodem **localmente** na máquina do usuário.

## 2. Veredito

**É viável.** A qualidade, porém, depende menos do modelo escolhido e mais de quanto do "professor" é implementado no código do app.

Um modelo local com o prompt "você é um professor nativo" se sai bem numa conversa casual. Como professor, ele falha: inventa regras gramaticais, não acompanha o nível do aluno e sai da persona. Essa estrutura pedagógica precisa morar na aplicação.

| Cenário | Viabilidade local |
|---|---|
| Chat por texto com correção e explicações | Alta |
| Conversa por voz no desktop (GPU ou Apple Silicon) | Alta |
| Avaliação de pronúncia por fonema | Média: possível, mas trabalhosa |
| Tudo local num celular comum | Baixa com qualidade de professor |

## 3. Componentes locais disponíveis

### 3.1 Modelo de linguagem (o "cérebro")

- **Qwen3 8B**: cerca de 7 GB de VRAM quantizado (Q4) e disponível no Ollama. Tem bom suporte a PT-BR e foi treinado em 119 idiomas.
- **Qwen3 14B**: cerca de 9 GB de VRAM (Q4), com mais nuance nas explicações.
- **Gemma 4** (variantes menores, em GGUF) e família Llama: alternativas, especialmente se as conversas forem só em inglês.
- Servidores: Ollama, llama.cpp ou vLLM.

Para o papel de professor com explicações em português, o recomendado é 14B ou maior. Modelos de 8B corrigem frases bem, mas explicam o *porquê* com menos confiabilidade.

### 3.2 Reconhecimento de fala (o "ouvido")

- **Whisper / faster-whisper**: padrão de mercado. O large-v3 cabe em cerca de 6 GB.
- **NVIDIA Parakeet TDT**: alternativa mais nova e rápida.
- Existem versões do Whisper treinadas só para inglês (`.en`), menores e mais precisas.

### 3.3 Síntese de voz (a "boca")

- **Kokoro-82M** (Apache 2.0): leve, usa 2 a 3 GB de VRAM e roda até em CPU. É a opção recomendada.
- **Qwen3-TTS**, **Chatterbox** e **XTTS**: alternativas mais pesadas ou com clonagem de voz.
- **Piper**: evitar como base, porque o repositório foi arquivado em outubro de 2025.

### 3.4 Pipeline de voz pronto

O pipeline open source `speech-to-speech` da Hugging Face, de julho de 2026, foi projetado para trocar cada componente de nuvem por um local. Ele combina Silero VAD, Parakeet ou Whisper, um modelo via llama.cpp ou vLLM e Kokoro ou Qwen3-TTS. É um bom ponto de partida em vez de montar a integração do zero.

## 4. Principais desafios

1. **Feedback de pronúncia.** O Whisper foi treinado para entender o que a pessoa *quis* dizer, então ele "corrige" a pronúncia errada sem avisar ("tink" vira "think"). Avaliar pronúncia exige um modelo de reconhecimento de fonemas (variantes do wav2vec2, por exemplo) com alinhamento forçado contra o texto esperado. É a parte com mais engenharia.
2. **Mistura de idiomas na voz.** Um professor bilíngue troca de idioma no meio da frase, e os modelos de voz pronunciam mal o trecho no outro idioma. Isso exige segmentar o texto por idioma e sintetizar cada trecho com a voz certa.
3. **Consistência pedagógica.** Modelos pequenos perdem o fio da conversa. A estrutura deve ficar no app:
   - perfil do aluno com nível CEFR (A1 a C2) e erros recorrentes, guardados em SQLite;
   - modos explícitos: conversa livre, correção, exercício e revisão;
   - repetição espaçada para vocabulário.
4. **Conhecimento contrastivo PT→EN.** Uma base curada de erros típicos de brasileiros, consultada via RAG, compensa muito as limitações de um modelo pequeno. Exemplos: falsos cognatos (*pretend*/pretender, *actually*/atualmente), decalques ("I have 30 years"), o som de *th* e a vogal extra no fim das palavras ("hot-chee-dogue"). **Este é o principal diferencial do produto.**
5. **Hardware e latência.** Com GPU de 8 a 12 GB ou Apple Silicon, o app roda bem. Em celular comum, só cabem modelos de 1 a 4B, abaixo do necessário para ensinar. Conversa por voz precisa responder em menos de 1 a 2 segundos para soar natural.

## 5. Variante: conversas 100% em inglês

### O que fica mais fácil

- **A voz fica monolíngue.** Não é preciso segmentar o texto por idioma, e as vozes em inglês são as melhores de qualquer modelo de síntese.
- **O reconhecimento de fala melhora.** Os modelos Whisper `.en` são menores, mais precisos e têm menos latência.
- **Modelos menores bastam.** Inglês é a língua mais forte de todos os modelos abertos. Um modelo de 4B a 8B explica gramática em inglês com boa confiabilidade, e a escolha passa a incluir todas as famílias de modelos.
- **O celular fica possível**, com concessões.
- **Prompt e testes ficam mais simples**, porque há uma língua a menos para validar.

### O que não muda

- A avaliação de pronúncia continua sendo a parte difícil.

### O que se perde

- **Iniciantes ficam de fora.** Alunos A1 e A2 não acompanham explicações em inglês, e o público passa a ser B1 em diante.
- **A persona perde parte do sentido.** O conhecimento contrastivo continua útil: o professor sabe que o aluno é brasileiro e antecipa os erros típicos, mas explica em inglês. Exemplo: *"Careful — 'pretend' doesn't mean 'pretender'. You want 'intend'."*
- **O aluno trava sem saída** quando não entende a explicação, e é aí que muita gente desiste.

### Meio-termo sugerido

Conversa e voz **100% em inglês**, com português apenas como **apoio em texto, sob demanda**: um botão "não entendi" que mostra a explicação em PT, ou uma tradução ao passar o mouse sobre as palavras. A síntese de voz continua monolíngue, a imersão fica como padrão e os iniciantes têm uma rede de segurança. O apoio pode ir sendo reduzido conforme o nível do aluno sobe.

## 6. Recomendações de arquitetura

- **Local por padrão, com nuvem opcional.** O app usa modelos locais e permite plugar uma API para quem quiser mais qualidade. Isso amplia o público sem abrir mão da privacidade.
- **A estrutura pedagógica fica no código**, não no prompt: perfil do aluno, modos e repetição espaçada.
- **Base de erros típicos de brasileiros** como ativo central do projeto, versionada no repo.
- **Verificar as licenças** de cada modelo (Qwen e Kokoro usam Apache 2.0), por se tratar de um projeto open source.

## 7. Proposta de MVP

**Fase 1**
- Desktop, com texto e voz.
- Qwen3 14B (ou um modelo menor, se tudo for em inglês), faster-whisper e Kokoro.
- Perfil do aluno, modos de conversa e correção.
- Base inicial de erros típicos de brasileiros (RAG).
- Conversa em inglês com apoio em PT por texto.

**Fase 2**
- Avaliação de pronúncia por fonemas.
- Repetição espaçada.
- Testes em hardware mais modesto e em celular.

## 8. Questões em aberto

- Público-alvo: incluir iniciantes (A1 e A2) ou começar do intermediário (B1)?
- Plataforma inicial: desktop nativo, web (WebGPU) ou ambos?
- Configuração mínima de hardware suportada.

## Fontes

- [Melhores LLMs locais para português do Brasil (2026) – PromptQuorum](https://www.promptquorum.com/pt/local-llms/best-local-llms-portuguese-language-2026)
- [Local Voice AI: Whisper, TTS & Offline Assistants (2026)](https://aivideosensei.com/guides/local-voice-whisper-offline-guide)
- [Best Local TTS Models 2026 – LocalAIMaster](https://localaimaster.com/blog/best-local-tts-models)
- [Build a Local Voice Assistant: The 2026 Speech-to-Speech Stack – LocalAIMaster](https://localaimaster.com/blog/local-speech-to-speech-assistant)
- [Local AI Voice Assistant Stack 2026 – Kunal Ganglani](https://www.kunalganglani.com/blog/local-ai-voice-assistant-whisper-piper-ollama)
