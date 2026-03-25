import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma_db"

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("Fel: OPENAI_API_KEY saknas i .env-filen.")
    raise SystemExit(1)

embedding_function = OpenAIEmbeddings(api_key=api_key)
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

model = ChatOpenAI(
    api_key=api_key,
    model="gpt-3.5-turbo",
    temperature=0
)

prompt_template = ChatPromptTemplate.from_template("""
Svara endast utifrån kontexten nedan.
Om svaret inte finns i kontexten, säg: "Jag hittar inte stöd för detta i dokumenten."

Kontext:
{context}

Fråga:
{question}
""")


def ask_question(query):
    try:
        results = db.similarity_search_with_relevance_scores(query, k=3)

        if len(results) == 0 or results[0][1] < 0.5:
            print("\nJag hittade inget tillräckligt relevant i dokumenten.\n")
            return

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

        prompt = prompt_template.format(context=context_text, question=query)
        response = model.invoke(prompt)

        print("\nSvar:")
        print(response.content)

        print("\nKällor:")
        for doc, score in results:
            source = doc.metadata.get("source", "Okänd källa")
            print(f"- {source} (relevans: {score:.2f})")

        print("\n" + "=" * 60 + "\n")

    except Exception as e:
        print(f"\nEtt fel uppstod vid sökning eller svarsgenerering: {e}\n")


def main():
    print(" RAG Chatbot körs i terminalen")
    print("Ställ en fråga baserat på dokumenten i databasen.")
    print("Skriv 'exit' för att avsluta.\n")

    while True:
        query = input("Skriv din fråga här: ")

        if query.lower() == "exit":
            print("Avslutar chatboten.")
            break

        if not query.strip():
            print("Du måste skriva en fråga.\n")
            continue

        ask_question(query)


if __name__ == "__main__":
    main()

# =========================
# Reflektion om verklig användning
# =========================
#
# Denna modell har implementerats som en textbaserad RAG-chatbot som körs i terminalen.
# Användaren kan ställa frågor om innehållet i de dokument som har lagrats i en Chroma-databas.
# Programmet söker först fram relevanta textdelar ur dokumenten och skickar därefter dessa som
# kontext till språkmodellen, som genererar ett svar. På detta sätt begränsas svaren till den
# information som faktiskt finns i dokumenten.
#
# Denna typ av modell hade kunnat användas i verkligheten som en intern kunskapsassistent i
# exempelvis ett företag, en myndighet, en skola eller en annan organisation. Istället för att
# användaren själv behöver läsa igenom flera dokument manuellt kan användaren ställa en fråga i
# naturligt språk och snabbt få ett sammanfattat svar baserat på dokumentens innehåll. En sådan
# lösning hade exempelvis kunnat användas för att söka i utbildningsplaner, policydokument,
# tekniska rapporter, interna riktlinjer eller andra större dokumentmängder.
#
# En viktig möjlighet med modellen är att den kan spara tid och göra information mer tillgänglig.
# Den kan också bidra till att användare hittar relevant information snabbare, särskilt när det
# finns många dokument att söka i. Genom att även visa källor kan användaren lättare kontrollera
# varifrån svaret kommer, vilket ökar transparensen i systemet.
#
# Samtidigt finns det flera utmaningar. Modellens svar blir bara så bra som de dokument som finns
# i databasen och hur väl sökningen lyckas hitta rätt textdelar. Om fel kontext hämtas kan svaret
# bli ofullständigt, otydligt eller missvisande trots att språkmodellen formulerar sig på ett
# övertygande sätt. En annan begränsning är att modellen inte själv avgör om dokumentens innehåll
# är aktuellt, korrekt eller uppdaterat. Om gamla eller bristfälliga dokument används finns en risk
# att användaren får fel information.
#
# Det finns även etiska aspekter att ta hänsyn till. En risk är att användaren litar för mycket på
# modellens svar utan att själv kontrollera källorna. Därför bör modellen användas som ett stöd för
# informationssökning och inte som den enda grunden för viktiga beslut. Om systemet används med
# interna dokument måste man också ta hänsyn till integritet, säkerhet och vem som har rätt att få
# tillgång till informationen. Om känsliga dokument används behöver systemet skyddas så att inte
# obehöriga kan få tillgång till innehållet.
#
# Det finns också affärsmässiga möjligheter med en sådan lösning. Ett företag eller en organisation
# skulle kunna använda modellen för att effektivisera intern support, onboarding, utbildning eller
# kunskapshantering. Det kan minska tiden som anställda lägger på att leta efter information och
# samtidigt göra det lättare att använda den kunskap som redan finns i organisationens dokument.
# Samtidigt kräver ett verkligt införande underhåll, uppdatering av dokumentdatabasen och testning
# för att säkerställa att svaren håller tillräckligt hög kvalitet.
#
# Om modellen skulle vidareutvecklas finns flera möjliga förbättringar. En förbättring hade varit
# att använda en mer avancerad utvärdering av både retrieval-delen och kvaliteten på svaren. En annan
# förbättring hade varit bättre chunking av dokumenten, så att rätt kontext hämtas mer träffsäkert.
# Modellen hade också kunnat kompletteras med ett grafiskt gränssnitt, exempelvis i Streamlit, för
# att bli mer användarvänlig. I denna uppgift valdes dock en terminalbaserad lösning, vilket är i
# linje med uppgiftsbeskrivningen där chatboten får köras antingen i terminalen eller som en
# Streamlit-applikation.
#
# =========================
# Utvärdering av chatboten
# =========================
#
# För att utvärdera chatboten kan man använda ett enkelt testsystem med ett antal frågor som är
# relevanta för dokumenten i databasen. Vid varje test kan man kontrollera:
#
# 1. Om chatboten hämtar relevanta dokumentdelar.
# 2. Om svaret faktiskt bygger på den hämtade kontexten.
# 3. Om svaret är relevant för frågan.
# 4. Om källorna som visas stämmer med dokumenten som använts.
# 5. Om chatboten avstår från att hitta på ett svar när stöd saknas i dokumenten.
#
# Exempel på testfrågor:
# - Vad handlar dokumentet om AI och IoT om?
# - Vad innehåller utbildningsplanen?
# - Vad tas upp i dokumentet om business intelligence?
#
# Ett bra resultat är att chatboten ger ett relevant svar, hänvisar till rätt källa och säger ifrån
# när tillräckligt stöd inte finns i dokumenten. På så sätt går det att göra en grundläggande
# bedömning av både systemets styrkor och dess begränsningar.