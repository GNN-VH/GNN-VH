
from utils import *

logs = open(log_file, 'w')
logs.write("Config:\n")
logs.write("LOSS_WEIGHT:{}\n".format(lamb))
logs.write("learning_rate:{}\n".format(learning_rate))
logs.write("rpn_per_img:{}\n".format(rpn_per_img))
logs.write("attention drop:{}\n".format(att_drop))
logs.write("feat_drop:{}\n".format(feature_drop))
logs.write("weight_drop:{}\n".format(dropout))
logs.write("ranking_rate{}\n".format(ranking_rate))
logs.write("leaky_alpha:{}\n".format(alpha))
logs.write("num_frame:{}\n".format(min_frame_per_clip))
logs.write("gamma:{}\n".format(l1_norm))
logs.write("Arch:{}".format(net.name))
logs.close()
print("{} {}".format(timestamp, gpu))

# data loader
best_acc = 0
count = 0
model_num = 0

for epoch in range(200):
    logs = open(log_file, 'a')
    cls_mean_loss = 0
    rk_mean_loss = 0

    t0 = time.time()
    net.train(mode=True)
    for feat, label, _, prop, vname in train_loader:
        if len(feat) != len(label):
            continue

        feat = [v.to(device) for v in feat]
        label = label[:max_frame]
        feat = feat[:max_frame]
        cls_loss, rk_loss, _ = loss_function(feat, label, vname)
        cls_mean_loss += cls_loss.detach().cpu().numpy()
        if rk_loss is not None:
            rk_mean_loss += rk_loss.detach().cpu().numpy()
        regular_loss = torch.FloatTensor([0]).to(device)
        for param in net.parameters():
            regular_loss += torch.sum(torch.abs(param))
        lamb = torch.FloatTensor([lamb]).to(device)
        optimizer.zero_grad()
        if rk_loss is not None:
            loss=lamb*cls_loss+(1-lamb)*rk_loss+l1_norm*regular_loss
        else:
            loss=lamb*cls_loss+l1_norm*regular_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
    dur.append(time.time() - t0)
    msg = "Epoch {:05d} | CLS Loss {:.4f}|RANK LOSS {:.4f}|Times(s) {:.4f}".format(
        epoch,
        cls_mean_loss / (train_dataset.__len__() - 2),
        rk_mean_loss / (train_dataset.__len__() - 2),
        np.mean(dur))
    print(msg)
    logs.write(msg + "\n")

    if (epoch ) % 3 == 0:
        bingo = 0
        all = 0
        cls_mean_loss = 0
        rk_mean_loss = 0
        rel_mean_loss = 0
        with torch.no_grad():
            net.eval()
            for feat, label, idx, prop, vname in test_loader:
                if len(feat) != len(label):
                    continue
                feat = [v.to(device) for v in feat]
                label = [l.to(device) for l in label]
                cls_loss, rk_loss, final_score = loss_function(feat, label, vname)
                cls_mean_loss += cls_loss.detach().cpu().numpy()
                if rk_loss is not None:
                    rk_mean_loss += rk_loss.detach().cpu().numpy()
                pos_predict_score, gt_score = frame_score_to_clip_score(final_score, idx, vname, reduce="avg")
                b, a = eval_clip(pos_predict_score, gt_score)
                bingo += b
                all += a
            msg = "Epoch {:05d} [TEST]| CLS Loss {:.4f} | RANK LOSS {:.4f}|Times(s) {:.4f}".format(
                epoch,
                cls_mean_loss / (test_dataset.__len__()),
                rk_mean_loss / (test_dataset.__len__()),
                np.mean(dur))
            print(msg)
            logs.write("{}\n".format(msg))
            acc = float(bingo) / float(all)
            print("acc:{}".format(float(bingo) / float(all)))
            logs.write("acc:{}\n".format(float(bingo) / float(all)))
            if acc > best_acc:
                if not os.path.exists(checkpoint):
                    os.makedirs(checkpoint)
                best_acc = acc
                torch.save(net.state_dict(), os.path.join(checkpoint, "model-{}.pt".format(model_num)))
                count = 0
                model_num += 1
            else:
                count += 1
                if count == 5:
                    print(best_acc)
                    break
    logs.close()
