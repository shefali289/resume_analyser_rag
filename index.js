import "dotenv/config";
import { readFile } from "node:fs/promises";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { createClient } from "@supabase/supabase-js";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { PromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

try {
  // Check for required environment variables
  const requiredEnvironmentVariables = [
    "HF_API_KEY",
    "SUPABASE_URL",
    "SUPABASE_KEY",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
  ];
  // Filter out the missing environment variables
  const missingEnvironmentVariables = requiredEnvironmentVariables.filter(
    (name) => !process.env[name]?.trim(),
  );

  // If any required environment variables are missing, throw an error
  if (missingEnvironmentVariables.length > 0) {
    throw new Error(
      `Missing required environment variable(s): ${missingEnvironmentVariables.join(", ")}. Add them to .env.`,
    );
  }

  // Read the contents of resume.txt
  const resumeText = await readFile(
    new URL("./resume.txt", import.meta.url),
    "utf8",
  );

  // Create chunks of the resume text of chunk size 500 and chunk overlap 50
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
  });

  // Split the resume text into chunks
  const docs = await splitter.createDocuments([resumeText]);

  // Create embeddings for the chunks
  const embeddings = new HuggingFaceInferenceEmbeddings({
    apiKey: process.env.HF_API_KEY,
    model: "sentence-transformers/all-MiniLM-L6-v2",
  });

  // Create Supabase client
  const client = createClient(
    process.env.SUPABASE_URL,
    process.env.SUPABASE_KEY,
  );

  // Store the documents and their embeddings in Supabase vector store - one-time
//   await SupabaseVectorStore.fromDocuments(
//     docs,
//     embeddings,
//     {
//       client,
//       tableName: "documents",
//       queryName: "match_documents",
//     },
//   );

  // Create a Supabase vector store instance to interact with the stored documents and embeddings
  const vectorStore = new SupabaseVectorStore(
    embeddings,
    {
      client,
      tableName: "documents",
      queryName: "match_documents",
    },
  );

  // Create a retriever from the vector store to retrieve relevant/matching documents based on a query
  const retriever = vectorStore.asRetriever();

  // Invoke the retriever with a query to get relevant documents
  const retrievedDocs = await retriever.invoke("What are my strengths based on this resume?");

  // Get the text content of the retrieved documents
  const context = retrievedDocs.map((doc) => doc.pageContent).join("\n\n");

  // Create Gemini llm instance
  const llm = new ChatGoogleGenerativeAI({
    apiKey: process.env.GEMINI_API_KEY,
    model: process.env.GEMINI_MODEL,
    temperature: 0.7,
  });

  // Create a prompt template to generate a response based on the retrieved context and the query
  const prompt = PromptTemplate.fromTemplate(
    `You are a helpful career coach. Based on the following resume context, give constructive feedback about the user's strengths, what kind of roles they are suitable for, and areas for improvement. Provide actionable advice to help the user improve their resume and career prospects based on the job market trends.
    Resume:
    {context}
    Feedback:`,
  );

  // Create a chain to process the prompt and generate a response using the llm
  const chain = prompt.pipe(llm).pipe(new StringOutputParser());
  const result = await chain.invoke({ context });
  console.log("Feedback based on the resume:\n", result);
} catch (error) {
  console.error("Application failed:", error.message);
  process.exitCode = 1;
}
