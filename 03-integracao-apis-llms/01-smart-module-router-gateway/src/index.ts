import { createServer } from "./server.ts";

const app = createServer();

await app.listen({ port: 3000, host: "0.0.0.0" });
console.log("server running at 3000");

app
  .inject({
    method: "POST",
    url: "/chat",
    body: { question: "hellou world" },
  })
  .then((response) => {
    console.log("Response status:", response.statusCode);
    console.log("Response body:", response.body);
  });
