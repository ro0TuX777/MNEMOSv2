import time

try:
    from llmlingua import PromptCompressor
except ImportError:
    PromptCompressor = None

class LLMLinguaAdapter:
    def __init__(self):
        if PromptCompressor:
            self.compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                use_llmlingua2=True
            )
        else:
            self.compressor = None

    def compress(self, context: str, target_ratio: float = 0.5) -> str:
        if not self.compressor: return context
        target_token = max(10, int(len(context.split()) * target_ratio))
        res = self.compressor.compress_prompt(
            context=[context],
            target_token=target_token,
            rank_method='longllmlingua'
        )
        return res.get("compressed_prompt", context)

class EchoFrameProcessor:
    def create_stable_packet(self, context: str) -> str:
        return f"[ECHO_FRAME_HEADER]\nGOVERNANCE: approval_required\n[FACTS]\nExtracted facts.\n[EVIDENCE]\n{context}\n[END_FRAME]"

class ModeRunner:
    def __init__(self):
        self.llm = LLMLinguaAdapter()
        self.echo = EchoFrameProcessor()

    def run_mode(self, mode: str, context: str) -> tuple:
        start = time.time()
        result = context
        if mode == "A":
            result = context
        elif mode in ["B", "F"]:
            result = self.echo.create_stable_packet(context)
        elif mode == "C":
            result = self.llm.compress(context)
        elif mode in ["D", "E"]:
            packet = self.echo.create_stable_packet(context)
            ev_start = packet.find("[EVIDENCE]") + len("[EVIDENCE]\n")
            ev_end = packet.find("[END_FRAME]")
            if ev_start != -1 and ev_end != -1:
                evidence = packet[ev_start:ev_end]
                comp_ev = self.llm.compress(evidence)
                result = packet[:ev_start] + comp_ev + packet[ev_end:]
            else:
                result = packet
        lat = (time.time() - start) * 1000
        return result, lat
