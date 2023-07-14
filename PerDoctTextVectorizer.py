# Implementação de um Tf-idf separado pelos documentos.
# Avalia a importancia de cada palavra a partir da quantidade de vezes que ela se repete dentro do documento.

from sklearn.feature_extraction.text import TfidfVectorizer

# O docs será considerado o nosso dataset
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
print(f)

# Verificar um documento por vez
for i, i2 in enumerate(docs):
    print(f"-------------------"
          f"Documento: {i + 1}\n"
          f"-------------------"
          )
    # Avalia a repetição de cada palavra dentro do documento em questão
    for y in tfidf_matrix[i].nonzero()[1]:#Avalia sob a frequencia utiliza dentro de cada documento
        word = f[y] # A paravra do documento
        tfidf = tfidf_matrix[i, y]
        print(f"{word}:{tfidf}")

