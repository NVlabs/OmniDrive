import torch
from typing import List, Optional

class Sampler:
    @staticmethod
    def sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        vocab_map: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        if temperature < 1e-3:
            top_k = 1
            top_p = 1.0
            logits = logits * 1000.0
        elif temperature != 1.0:
            logits = logits / temperature

        batch_size, vocab_size = logits.shape
        selected_indices = None
        
        if top_k > 0:
            k = min(top_k, vocab_size)
            k_values, k_indices = torch.topk(logits, k=k, dim=-1)
            
            if top_p < 1.0:
                probs = torch.softmax(k_values, dim=-1)
                cumsum = torch.cumsum(probs, dim=-1)
                
                r = torch.rand((batch_size, 1), device=logits.device, generator=generator) * top_p
                
                mask = cumsum >= r
                local_indices = torch.argmax(mask.int(), dim=-1, keepdim=True) 
                
                selected_indices = torch.gather(k_indices, 1, local_indices)
            else:
                probs = torch.softmax(k_values, dim=-1)
                local_indices = torch.multinomial(probs, num_samples=1, generator=generator)
                selected_indices = torch.gather(k_indices, 1, local_indices)
                
        elif top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            
            probs = torch.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(probs, dim=-1)
            
            r = torch.rand((batch_size, 1), device=logits.device, generator=generator) * top_p
            
            mask = cumsum >= r
            local_indices = torch.argmax(mask.int(), dim=-1, keepdim=True)
            
            selected_indices = torch.gather(sorted_indices, 1, local_indices)
            
        else:
            probs = torch.softmax(logits, dim=-1)
            selected_indices = torch.multinomial(probs, num_samples=1, generator=generator)

        if vocab_map is not None:
            flat_indices = selected_indices.view(-1)
            mapped_indices = vocab_map[flat_indices.long()]
            selected_indices = mapped_indices.view(batch_size, 1)

        return selected_indices

    @staticmethod
    def update_sequence(
        output_ids: List[List[int]],
        finished: List[bool],
        selected_token_ids: torch.Tensor,
        eos_id: int
    ) -> int:
        selected_ids_cpu = selected_token_ids.flatten().cpu().tolist()
        
        batch_size = len(output_ids)
        num_newly_finished = 0
        
        for i in range(batch_size):
            if not finished[i]:
                token_id = selected_ids_cpu[i]
                output_ids[i].append(token_id)
                
                if token_id == eos_id:
                    finished[i] = True
                    num_newly_finished += 1
                    
        return num_newly_finished
