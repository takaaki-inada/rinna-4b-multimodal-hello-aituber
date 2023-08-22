import { Message } from "../messages/messages";

export async function getLocalChatResponseStream(
  text: string,
) {

  const res = await fetch('http://localhost:8000/generate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      'text': text,
      // https://developers.rinna.co.jp/ja-jp/pricing
      // 1文章、60文字
      'min_length': 6,
      'max_new_tokens': 32,
    }),
  })
  const reader = res.body?.getReader();
  if (res.status !== 200 || !reader) {
    throw new Error("Something went wrong");
  }

  const stream = new ReadableStream({
    async start(controller: ReadableStreamDefaultController) {
      const decoder = new TextDecoder("utf-8");
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const data = decoder.decode(value);
          controller.enqueue(data);
        }
      } catch (error) {
        controller.error(error);
      } finally {
        reader.releaseLock();
        controller.close();
      }
    },
  });

  return stream;
}

export async function getLocalChatStatusResponseStream() {
  const res = await fetch('http://localhost:8000/status', {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
  })
  const reader = res.body?.getReader();
  if (res.status !== 200 || !reader) {
    throw new Error("Something went wrong");
  }

  const stream = new ReadableStream({
    async start(controller: ReadableStreamDefaultController) {
      const decoder = new TextDecoder("utf-8");
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const data = decoder.decode(value);
          controller.enqueue(data);
        }
      } catch (error) {
        controller.error(error);
      } finally {
        reader.releaseLock();
        controller.close();
      }
    },
  });

  return stream;
}
