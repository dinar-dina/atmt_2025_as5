import torch
import sentencepiece as spm
from seq2seq.models import Seq2SeqModel

def decode(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, max_out_len: int,
           tgt_tokenizer: spm.SentencePieceProcessor, args, device: torch.device):
    """Decodes a sequence without teacher forcing. Works by relying on the model's own predictions, rather than the ground truth (trg_)"""
    batch_size = src_tokens.size(0)
    BOS = tgt_tokenizer.bos_id()
    EOS = tgt_tokenizer.eos_id()
    PAD = tgt_tokenizer.pad_id()
    generated = torch.full((batch_size, 1), BOS, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # (batch, 1, 1, max_out_len)
    full_pad_mask = torch.zeros(batch_size, 1, 1, max_out_len, dtype=torch.bool, device=device)

    for t in range(max_out_len):
        # Slice only what is needed: fast operation
        seq_len = generated.size(1)
        trg_pad_mask = full_pad_mask[:, :, :, :seq_len].clone()
        trg_pad_mask[:, :, :, :] = (generated == PAD).unsqueeze(1).unsqueeze(2)

        # Forward pass: use only the generated tokens so far
        output = model(src_tokens, src_pad_mask, generated, trg_pad_mask)
        # Get the logits for the last time step
        next_token_logits = output[:, -1, :]  # last time step
        next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)  # greedy

        # Append next token to each sequence
        generated = torch.cat([generated, next_tokens], dim=1)

        # Mark sequences as finished if EOS is generated
        finished = finished | (next_tokens.squeeze(1) == EOS)
        if finished.all():
            break
    # Remove initial BOS token and anything after EOS
    predicted_tokens = []
    for seq in generated[:, 1:].tolist():
        if EOS in seq:
            idx = seq.index(EOS)
            seq = seq[:idx+1]
        predicted_tokens.append(seq)
    return predicted_tokens

def beam_search_decode(model: Seq2SeqModel, src_tokens: torch.Tensor, src_pad_mask: torch.Tensor, max_out_len: int,
                       tgt_tokenizer: spm.SentencePieceProcessor, args, device: torch.device, beam_size: int = 5, alpha: float = 0.7):
    """Beam Search decoding compatible with Transformer-based Seq2Seq models."""
    model.eval()
    BOS, EOS, PAD = tgt_tokenizer.bos_id(), tgt_tokenizer.eos_id(), tgt_tokenizer.pad_id()
    # __QUESTION 1: what does this line set up and why is the beam represented this way?
    # The beam is initialized as a list containing a sequence starting only with the BOS token and an initial score of 0. Each beam entry stores a pair(sequence, score) because beam search must track both the growing output tokens and the cumulative log-probability of that partial translation. Representing the beam as a list allows us to maintain multiple competing hypotheses simultaneously and to expand and re-rank them at every decoding step. 
    beams = [(torch.tensor([[BOS]], device=device), 0.0)]
    
    # Pre-allocate max-size pad mask (batch=1 for decoding)
    full_pad_mask = torch.zeros(1, 1, 1, max_out_len, dtype=torch.bool, device=device)

    for _ in range(max_out_len):
        new_beams = []
        for seq, score in beams:
            if seq[0, -1].item() == EOS:
                new_beams.append((seq, score))
                continue

            seq_len = seq.size(1)
            trg_pad_mask = full_pad_mask[:, :, :, :seq_len].clone()
            trg_pad_mask[:, :, :, :] = (seq == PAD).unsqueeze(1).unsqueeze(2)

            with torch.no_grad():
                logits = model(src_tokens, src_pad_mask, seq, trg_pad_mask)[:, -1, :]
                
                # __QUESTION 2: Why do we need to create trg_pad_mask here and how does it affect the model's predictions?
                # We need the pad mask to make the decoder does not attend to padding positions, which would introduce noise into the attention computation. If padding tokens are left unmasked, they will distort the probability and lead to unreliable next-token prediction.
                # __QUESTION 3: Explain the purpose of applying log_softmax and selecting top-k tokens here.
                # we apply log_softmax because beam search accumulates sequence scores by adding log-probabilities instead of multiplying raw softmax probabilities, which would quickly underflow toward 0. Taking the top-k log-probabilities selectes only the most promising next tokens, preventing the search space from exploding and allowing the decoder to explore the best one. 
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=-1)

            for k in range(beam_size):
                # __QUESTION 4: explain the tensor shapes and the logic when creating new_seq and new_score below. Is any broadcasting or indexing issue possible?
                # when expending a beam, seq has shape(1,L), while topk_ids [: , k] has shape (1,). We use unsqueeze (0)to turn it into (1,1) so that its batch and time dim align with seq, allowing torch.cat to append the new token without any broadcasting issues. The new_score is computed by adding a float(score) to another float obtained from topk_log_probs[: , k] .item(), so no shape or indexing conflicts occur.
                new_seq = torch.cat([seq, topk_ids[:, k].unsqueeze(0)], dim=1)
                raw_score = score + topk_log_probs[:, k].item()
                seq_len = new_seq.size(1)
                length_penalty = ((5 + seq_len)**alpha) / (6**alpha)
                new_score = raw_score / length_penalty
                new_beams.append((new_seq, new_score))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        # __QUESTION 5: Why do we check for EOS here and what does it imply for beam search?
        #we check for EOS because it marks the completed sequence that should no longer ne expended. If every beam hypothesis ends with the EOS token, the search can safely stop, since no further extensions will improve the final translation. And it is also treated as the terminal state in the beam search process.
        if all(seq[0, -1].item() == EOS for seq, _ in beams):
            break
    best_seq, _ = beams[0]
    # __QUESTION 6: What is returned, and why are we squeezing, converting to list and wrapping in another list here?
    #it returns a batch list containing one decoded token list. .squeeze() removed the unnecessary batch dimension from the sequence tensor, and .tolist() converts the tensor into a normal python list of token IDs. The result is wrapped in another list preserve the standard ‘list of sequences’ output format, even though beam search runs with a batch size of 1.
    return [best_seq.squeeze(0).tolist()]

def beam_search_relative_pruning(model, src_tokens, src_pad_mask, max_out_len,
                                 tgt_tokenizer, args, device,
                                 beam_size=5, alpha=0.7, tau_r=-3.0):

    BOS, EOS, PAD = tgt_tokenizer.bos_id(), tgt_tokenizer.eos_id(), tgt_tokenizer.pad_id()
    beams = [(torch.tensor([[BOS]], device=device), 0.0)]
    full_pad_mask = torch.zeros(1,1,1,max_out_len, dtype=torch.bool, device=device)

    for _ in range(max_out_len):
        candidates = []

        for seq, score in beams:
            if seq[0,-1].item() == EOS:
                candidates.append((seq, score))
                continue

            seq_len = seq.size(1)
            trg_pad_mask = full_pad_mask[:,:,:,:seq_len].clone()
            trg_pad_mask[:,:,:] = (seq==PAD).unsqueeze(1).unsqueeze(2)

            with torch.no_grad():
                logits = model(src_tokens, src_pad_mask, seq, trg_pad_mask)[:,-1,:]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            topk_log_probs, topk_ids = log_probs.topk(beam_size, dim=-1)

            for k in range(beam_size):
                new_seq = torch.cat([seq, topk_ids[:,k].unsqueeze(0)], dim=1)
                raw_score = score + topk_log_probs[:,k].item()
                seq_len = new_seq.size(1)
                length_penalty = ((5 + seq_len)**alpha) / (6**alpha)
                new_score = raw_score / length_penalty
                
                candidates.append((new_seq, new_score))

        if not candidates:
            break

        best_score = max(score for (_, score) in candidates)
        pruned_beams = [
            (seq, score) for (seq, score) in candidates 
            if score >= best_score + tau_r  
        ]

        beams = sorted(pruned_beams, key=lambda x: x[1], reverse=True)[:beam_size]

        if all(seq[0,-1].item() == EOS for seq, _ in beams):
            break

    return [beams[0][0].squeeze(0).tolist()]

def beam_search_node_pruning(model, src_tokens, src_pad_mask, max_out_len,
                             tgt_tokenizer, args, device,
                             beam_size=5, alpha=0.7, max_per_node=2):

    BOS, EOS, PAD = tgt_tokenizer.bos_id(), tgt_tokenizer.eos_id(), tgt_tokenizer.pad_id()
    beams = [(torch.tensor([[BOS]], device=device), 0.0)]
    full_pad_mask = torch.zeros(1,1,1,max_out_len, dtype=torch.bool, device=device)

    for _ in range(max_out_len):
        new_beams = []

        for seq, score in beams:
            if seq[0,-1].item() == EOS:
                new_beams.append((seq, score))
                continue

            seq_len = seq.size(1)
            trg_pad_mask = full_pad_mask[:,:,:,:seq_len].clone()
            trg_pad_mask[:,:,:] = (seq==PAD).unsqueeze(1).unsqueeze(2)

            with torch.no_grad():
                logits = model(src_tokens, src_pad_mask, seq, trg_pad_mask)[:,-1,:]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            topk_log_probs, topk_ids = log_probs.topk(max_per_node, dim=-1)

            # ---- node pruning: keep only top max_per_node extensions ----
            for k in range(max_per_node):
                new_seq = torch.cat([seq, topk_ids[:,k].unsqueeze(0)], dim=1)
                raw_score = score + topk_log_probs[:,k].item()
                seq_len = new_seq.size(1)
                length_penalty = ((5 + seq_len)**alpha) / (6**alpha)
                new_score = raw_score / length_penalty
                
                new_beams.append((new_seq, new_score))

        if not new_beams:
            break

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

        if all(seq[0,-1].item() == EOS for seq, _ in beams):
            break

    return [beams[0][0].squeeze(0).tolist()]