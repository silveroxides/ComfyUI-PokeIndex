from comfy_api.latest import ComfyExtension, IO
from .poke_index import PokeIndex

class PokeIndexExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [PokeIndex]

async def comfy_entrypoint() -> PokeIndexExtension:
    return PokeIndexExtension()


__all__ = ['comfy_entrypoint']