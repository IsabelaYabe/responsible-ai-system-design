# IA para Compreensão Textual

### Problema

Compreender um texto não é apenas sobre conhecer palavras; é uma tarefa que envolve conectar conceitos, lembrar de referências, identificar contextos culturais e integrar informações novas com conhecimento pré-existente do próprio leitor. O processo de ler envolve suas próprias dificuldades, já que o leitor vai frequentemente encontrar termos desconhecidos, passagens densas ou referências culturais novas – precisando então sair do texto para procurar explicações em outros meios ou fontes.

Ferramentas atuais consideram esse problema de forma reativa e isolada. Buscas em um dicionário retornam uma definição descolada do contexto; enquanto uma busca em ferramentas de pesquisa como o Google retorna resultados soltos, que não consideram o que estava sendo lido, em qual ponto do texto o leitor estava, ou suas dúvidas passadas. Mesmo assistentes de IA, quando abertos em uma janela separada, forçam o leitor a apresentar o seu contexto do zero.

Essa fragmentação é particularmente aguda quando o texto lido é técnico, acadêmico, literário ou está escrito em uma língua estrangeira – precisamente porque são textos que exigem mais do leitor. Essa troca entre um texto primário e recursos externos aumenta a carga cognitiva e piora tanto a imersão quanto a retenção do leitor.

**Pergunta central de design:** como sistemas de IA podem apoiar a compreensão textual de forma personalizada sem substituir o processo ativo de interpretação do usuário-leitor?

### Trabalhos passados

No artigo "Reading.help: Supporting EFL Readers with Proactive and On-Demand Explanation of English Grammar and Semantics" (Chung et al., 2025), é apresentada uma arquitetura para um assistente digital de leitura focado em um contexto de aprendizado de inglês como língua estrangeira (EFL, *English as a Foreign Language*).

### Proposta

Nossa solução é um assistente de IA para leitura capaz de oferecer apoio contextual sob demanda, integrado à experiência digital de leitura.

Quando um leitor seleciona uma palavra, frase ou trecho de um texto, o sistema gera conteúdo de apoio com base no conteúdo que está sendo lido. Ao invés de uma resposta genérica, a geração do sistema é baseada nos trechos em torno da seleção do leitor, nos temas da obra ou artigo e nas interações passadas do leitor durante aquela sessão de leitura.

As modalidades de auxílio do sistema incluem:

* Explicações de termos e conceitos
* Contextualização histórica, cultural ou temática
* Simplificação de trechos complexos
* Paráfrases e exemplos práticos
* Retorno de referências passadas dentro da mesma obra
* Conexões entre conceitos relacionados
* Diagramas e exemplos visuais

Dois princípios centrais ao sistema são os de autonomia do leitor e o de profundidade progressiva. O primeiro está no fato de que o sistema só age quando iniciado pelo leitor, de forma reativa e não proativa. O objetivo do sistema é reduzir o atrito causado por interrupções na leitura; logo ele só pode ser ativado a partir de uma intenção explícita. A partir daí segue o segundo princípio: a ideia do sistema não é limitar o processo cognitivo de interpretação do texto, mas auxiliá-lo. Respostas iniciais são pequenas e baseadas no pedido do leitor; o leitor pode então pedir mais explicações ou respostas a partir dessa geração, gerando um raciocínio maior e mais detalhado a partir dessas interações.

### *Casos de uso*

## Arquitetura

### Mecanismo anti-spoiler

Para