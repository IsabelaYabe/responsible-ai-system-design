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

Tomamos como persona central **Júlia**, leitora brasileira de inglês intermediário que lê romances no idioma original por prazer e prática (detalhamento em `docs/user_journey.md`). Seus pontos de fricção típicos motivam os casos de uso:

* **Entender uma palavra ou frase difícil** — Júlia seleciona um termo e recebe uma definição/paráfrase ancorada no contexto do trecho, sem precisar sair do app.
* **Compreender uma passagem densa** — uma simplificação ou explicação do trecho selecionado.
* **Relembrar algo já lido** — "quem era esse personagem?" retorna menções anteriores *dentro da obra*, limitadas ao ponto em que ela está.
* **Contextualizar** — contexto histórico, cultural ou temático sob demanda.

Em todos os casos valem os princípios de autonomia (o sistema só age quando acionado) e de profundidade progressiva (resposta curta primeiro, aprofundável). E em todos eles a ajuda é limitada ao que já foi lido — é o mesmo mecanismo anti-spoiler que serve tanto à contextualização quanto ao "relembrar".

## Arquitetura

O fluxo central do sistema é um *pipeline* de recuperação-aumentada (RAG, *Retrieval-Augmented Generation*) adaptado ao contexto de leitura. A obra que está sendo lida é segmentada em *chunks*, cada um anotado com a sua **posição** no texto (no nosso protótipo, o índice do capítulo). A partir de uma seleção ou pergunta do leitor, o sistema (i) recupera os trechos mais relevantes da obra, (ii) aplica um **filtro de posição** que descarta qualquer trecho além do ponto em que o leitor se encontra e (iii) gera a resposta condicionada apenas aos trechos recuperados. O diagrama do protótipo (`notebooks/poc_reading_assistant_architecture.svg`) detalha esse fluxo.

A escolha por RAG, e não por simplesmente enviar a obra inteira ao modelo, não é só uma questão de custo ou de limite de contexto: ela é o que torna a garantia anti-spoiler **estrutural**. O modelo não recebe — e portanto não pode revelar — aquilo que o filtro de recuperação não deixou passar.

### Mecanismo anti-spoiler

**O problema do spoiler é, antes de tudo, um problema de posição.** Um mesmo fato sobre uma obra pode ser uma explicação legítima ou um spoiler, dependendo unicamente de onde o leitor está. "Quem é o senhor Wickham?" é uma pergunta cuja resposta é segura no capítulo 15 de *Orgulho e Preconceito*, mas que envolve revelações se respondida com base em capítulos posteriores. Logo, o mecanismo precisa raciocinar sobre a posição do leitor, e não apenas sobre o conteúdo da pergunta.

**A ideia central é amarrar a fronteira de conhecimento à camada de recuperação, e não ao *prompt*.** Pedir a um modelo de linguagem que "não dê spoilers" é uma salvaguarda frágil: depende de o modelo conhecer a obra, lembrar onde o leitor está e resistir à tentação de completar a resposta — exatamente o tipo de instrução proibitiva que tende a falhar (o efeito de *ironic rebound*, em que proibir a menção de algo o torna mais saliente; Zhou et al., 2023). No nosso desenho, a fronteira é imposta **antes** da geração, por um filtro determinístico sobre os metadados de posição dos *chunks*:

> recupera-se um excedente de candidatos; descartam-se todos os *chunks* cujo índice de capítulo seja maior que a posição do leitor; usam-se apenas os sobreviventes.

Essa única operação é a parte que sustenta o mecanismo: o modelo *fisicamente não vê* o texto posterior, independentemente do que o *prompt* diga.

O protótipo recupera por dois caminhos, ambos sujeitos ao mesmo filtro de posição:

* **Recuperação por entidade (lexical):** um modelo identifica se a pergunta gira em torno de uma entidade nomeada (personagem, lugar, conceito). Em caso positivo, busca-se essa entidade por correspondência textual, distribuindo os resultados ao longo dos capítulos já lidos — útil para perguntas do tipo "quem é X?", em que a recall por nome importa mais que a similaridade semântica.
* **Recuperação semântica (embeddings):** para as demais perguntas, usa-se busca vetorial (FAISS, cosseno) sobre *embeddings* dos trechos.

Na prática, o produto não opera sobre perguntas livres, mas sobre uma **seleção + intenção**: o leitor destaca um trecho e escolhe o que quer dele (*definir*, *parafrasear*, *contextualizar* ou *relembrar*). Um **roteador de intenção** (*dispatcher*) mapeia cada intenção para a estratégia de recuperação adequada — todas sujeitas ao mesmo filtro de posição:

* **Definir** e **contextualizar** usam recuperação semântica escopada ao que já foi lido;
* **Parafrasear** opera apenas sobre o trecho selecionado, sem recuperação;
* **Relembrar** usa uma **coleta exaustiva**: em vez de poucos trechos representativos (como na recuperação por entidade do QA, que limita as menções por capítulo), reúne *todas* as menções anteriores da entidade dentro do que já foi lido, em ordem cronológica. Completude importa mais que parcimônia quando o objetivo é reconstruir "quem era esse personagem?". A limitação conhecida é que a correspondência é textual — apelidos e referências pronominais escapam —, mas isso é uma questão de cobertura, não de spoiler: o filtro de posição continua valendo.

A **geração** usa um *prompt* de enquadramento positivo e escopado ao contexto (Zhou et al., 2023): o modelo é instruído sobre o que ele *é* — um companheiro de leitura que só conhece o que já foi lido — em vez de receber uma lista de proibições. O *prompt* também define um **espectro graduado de resposta** — responder integralmente, responder parcialmente sinalizando o que ainda falta, ou recusar — de modo a evitar tanto o spoiler quanto a recusa excessiva. Vale notar que o *prompt* apenas molda o tom e o comportamento; a garantia dura continua sendo o filtro de recuperação.

### Protótipo interativo

Além do *harness* de avaliação (um *notebook*), construímos um protótipo interativo (FastAPI + JavaScript) que modela a interação real do produto. O texto é renderizado **apenas até a posição do leitor**, controlada por um *slider*, de modo que só é possível selecionar aquilo que já foi "lido"; o leitor destaca um trecho, escolhe uma intenção e recebe a resposta já com a fronteira de posição aplicada. Mover o *slider* faz spoilers aparecerem ou desaparecerem — uma demonstração direta e tangível do mecanismo. O protótipo e o *notebook* são duas camadas finas sobre o mesmo *back-end* de recuperação: respondem a perguntas diferentes (demonstrar vs. medir), mas compartilham o filtro de posição.

### Trabalhos relacionados e ausência de *benchmark*

A detecção de spoilers é um problema estudado em PLN, mas com um formato diferente do nosso. Os dois conjuntos de dados de referência — Goodreads (Wan et al., 2019) e TV Tropes (Chang et al., 2021) — rotulam a "spoilerosidade" de uma sentença de forma **global** (esta frase de uma resenha revela um ponto importante da obra como um todo?), e não **relativa à posição do leitor**. Além disso, são tarefas de *classificação sobre texto existente*, não de *geração/resposta livre de spoilers*. Por isso, **não há *benchmark* pronto que possamos usar** para a nossa tarefa: precisamos construir o nosso próprio conjunto de avaliação. O parente mais próximo do nosso desenho são sistemas conscientes de posição como o X-Ray Recaps (resumos de episódios/temporadas com salvaguardas) e pesquisas sobre "resumos dinâmicos por progresso de leitura" — todos voltados a *sumarização*, ao passo que a nossa contribuição é **resposta sob demanda (QA) com a fronteira imposta na camada de recuperação**.

### Validação

A avaliação compara duas configurações que diferem **exclusivamente pelo filtro de posição** (mantendo idêntica a estratégia de recuperação, para isolar o efeito do mecanismo):

* **Bounded** (com filtro): só enxerga capítulos até a posição do leitor.
* **Unbounded** (*baseline*, sem filtro): enxerga a obra inteira — representa o problema que estamos resolvendo.

O conjunto de avaliação é gerado por LLM (e revisado manualmente) em três níveis de risco de spoiler: **safe** (resposta contida no que já foi lido), **boundary** (no limite da posição atual) e **spoiler** (resposta só aparece adiante). O comportamento esperado é: no nível *spoiler*, o *bounded* deve recusar/responder parcialmente enquanto o *unbounded* tende a vazar; no nível *safe*, ambos devem responder.

A classificação de cada resposta é feita por um **LLM-como-juiz**, que emite um de cinco veredictos: `safe_full`, `safe_partial`, `safe_defer` (os três comportamentos corretos), `over_refusal` (recusa indevida) e `leak` (vazamento de spoiler). As métricas principais são a **taxa de vazamento** (proporção de `leak`), a **taxa de recusa excessiva** (`over_refusal`) e a **taxa de comportamento correto**, reportadas por nível e por configuração.

O uso de LLM-como-juiz tem limitações conhecidas, que endereçamos explicitamente:

* **Viés de auto-preferência:** modelos tendem a favorecer as próprias gerações. Por isso o juiz usa um modelo **diferente e mais forte** (Sonnet) do que o que gera as respostas (Haiku).
* **Concordância com humanos:** a confiabilidade do juiz só é defensável se ancorada em rótulos humanos. O *pipeline* prevê uma coluna de rótulo humano e calcula o **kappa de Cohen** entre juiz e humano sobre o conjunto avaliado; o *prompt* do juiz também inclui exemplos (*few-shot*) para melhorar a concordância.

### Avaliação por intenção

A avaliação descrita acima exercita um único *prompt* genérico de QA. Mas, como o produto é seleção + intenção, estendemos o *harness* para testar **cada intenção no seu próprio caminho de recuperação**, medindo **dois eixos ortogonais**:

* **Segurança** (o mesmo eixo anti-spoiler acima, com os cinco veredictos): aplicado às intenções sensíveis à posição — **contextualizar** e **relembrar** —, comparando *bounded* vs. *unbounded*. São justamente as intenções em que recuperar bem tenta o modelo a puxar material posterior, e onde o contraste com o *baseline* sem filtro carrega a tese anti-spoiler.
* **Qualidade** (um juiz separado, com rótulos `good`/`partial`/`poor`): aplicado a todas as intenções avaliadas, sobre a resposta *bounded* (a do produto). Os eixos são independentes e medidos separadamente: um vazamento pode ser bem escrito, e uma resposta perfeitamente segura pode ser inútil.

**Definir** é avaliado apenas em qualidade: seu modo de falha é correção (a definição está certa para o sentido em uso?), e não posição — por isso não tem nível de spoiler nem contraste *unbounded*; seus itens de avaliação são simplesmente as seleções (palavras/expressões), sem a divisão em níveis. **Parafrasear** fica fora do escopo desta avaliação, por ser tratado como um componente à parte, com métricas próprias.

Como a rotulagem humana ainda não foi feita, o *harness* já expõe os **dois espaços de rótulo lado a lado** e calcula o **kappa de Cohen para cada eixo** — segurança *e* qualidade —, de modo que nenhuma das métricas seja confiada sem a sua própria âncora humana.

### Limitações

O protótipo valida a ideia em **uma única obra** (*Orgulho e Preconceito*), em **uma única posição** de leitor (capítulo 15) e com um conjunto pequeno de perguntas — suficiente como prova de conceito, mas não para afirmações estatísticas fortes. A granularidade da posição é o capítulo (não o parágrafo/sentença). E, embora o filtro de recuperação seja determinístico, a *avaliação* depende da qualidade dos juízes — agora dois, um para segurança e outro para qualidade —; daí a importância da ancoragem humana (kappa de Cohen) em ambos os eixos.