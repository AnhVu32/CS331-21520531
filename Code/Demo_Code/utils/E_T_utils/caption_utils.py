import torch

def generate_caption(model, word_to_idx, idx_to_word, img_emb, max_sequence_len, start_token, beam_size, pad_token, device):
    # img_emb = img_emb.permute(0, 2, 3, 1).reshape(img_emb.size(0), -1, img_emb.size(3)).to(device) -> lỗi liên quan đến việc tạo bản copy tensor
    img_emb = img_emb.permute(0, 2, 3, 1).view(img_emb.size(0), -1, img_emb.size(1)).to(device)

    caption = []
    seq = [pad_token] * max_sequence_len
    seq[0] = start_token
    seq = torch.tensor(seq).squeeze(0).view(1, -1).to(device)
    beams = [[start_token]]
    beam_scores = [0.0]

    for _ in range(max_sequence_len - 1):
        new_beams = []
        new_beam_scores = []

        for beam, beam_score in zip(beams, beam_scores):
            seq = [pad_token] * max_sequence_len
            seq[:len(beam)] = beam
            seq = torch.tensor(seq).squeeze(0).view(1, -1).to(device) # sửa ở đây
            
            out, _ = model(seq, img_emb)
            pred = out[len(beam) - 1, 0, :]
            top_k_tokens = torch.topk(pred, beam_size).indices.tolist()
            top_k_scores = torch.topk(pred, beam_size).values.tolist()

            for token, score in zip(top_k_tokens, top_k_scores):
                new_beam = beam + [token]
                new_beam_score = beam_score + score
                new_beams.append(new_beam)
                new_beam_scores.append(new_beam_score)

        top_k_indices = torch.topk(torch.tensor(new_beam_scores), beam_size).indices.tolist()
        beams = [new_beams[index] for index in top_k_indices]
        beam_scores = [new_beam_scores[index] for index in top_k_indices]

    best_beam = beams[0]
    caption = [idx_to_word[token] for token in best_beam[1:]]
    sentence = []
    for word in caption:
        if word == '<end>':
            break
        sentence.append(word)
    sentence = ' '.join(sentence)
    return sentence
