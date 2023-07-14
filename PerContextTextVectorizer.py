from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "Eleições presidenciais: Os candidatos debatem suas propostas em busca do voto popular dos transporte publico.",
    "Reforma da previdência: Mudanças no sistema de aposentadoria geram polêmica e protestos.",
    "Corrupção: Escândalos revelam desvio de verbas públicas e abalam a confiança na classe política.",
    "Crise migratória: O aumento do fluxo de imigrantes gera desafios políticos e humanitários.",
    "Polarização ideológica: Divisões profundas entre grupos políticos dificultam o diálogo e a governabilidade.",
    "Medidas de combate à pandemia: Decisões políticas impactam a saúde pública e a economia.",
    "Acesso à educação: Debates sobre políticas públicas visam garantir uma educação de qualidade para todos.",
    "Copa do Mundo: As melhores seleções competem pelo título em uma atmosfera de emoção e rivalidade.",
    "Transferências milionárias: Jogadores de futebol são contratados por valores astronômicos, movimentando o mercado.",
    "Clássicos regionais: Partidas entre times rivais acirram a rivalidade e empolgam torcedores.",
    "VAR: O uso do árbitro de vídeo gera debates sobre sua eficácia e impacto no jogo.",
    "Jogadores ídolos: Atletas carismáticos e talentosos conquistam a admiração dos fãs.",
    "Campeonatos nacionais: Disputas acirradas pelos títulos movimentam o calendário futebolístico.",
    "Futebol feminino: A modalidade ganha cada vez mais visibilidade e apoio em todo o mundo.",
    "Inteligência Artificial: Avanços no campo da IA estão transformando diversos setores, como saúde e transporte."
]
vectorizer = TfidfVectorizer()

# Calcula a importancia de nossas palavras dentro do docto e/ou em um contexto geral
tfidf_matrix = vectorizer.fit_transform(docs)

# Obtem uma lista com as palavras dentro de todos os nossos doctos
f = vectorizer.get_feature_names_out()

"""
Cálculo da média dos valores TF-IDF para cada recurso a partir de todos as palavras nos doctos

-----------------------------------

Essa verificação vai ser feita através da variavel AXIS de 0..N é definido a coluna a qual 
nós iremos fazer a verificação

-----------------------------------
Quando for fazer a conexão com o banco de dados então dar uma atenção aqui e não esquecer
de trocar o axis de verificação

O codigo ira ficar mean_tfidf = tfidf_matrix.mean(axis=n)
-----------------------------------
"""
mean_tfidf = tfidf_matrix.mean(axis=0)

# Exibindo os recursos e suas médias de TF-IDF
for idf, namef in enumerate(f):
    tfidf_mean_value = mean_tfidf[0, idf]
    print(f"{namef}: {tfidf_mean_value}")