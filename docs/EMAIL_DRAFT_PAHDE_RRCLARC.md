# Draft email to Pahde et al. (not sent; the author sends it)

**To:** wojciech.samek@hhi.fraunhofer.de, sebastian.lapuschkin@hhi.fraunhofer.de (corresponding authors, paper p. 1)
**Cc:** Frederik Pahde (first author; address to be added)
**Subject:** Question about the RR-ClArC configuration for ISIC2019 (Table 1), "Ensuring Medical AI Safety"

Dear Dr. Samek, Dr. Lapuschkin and Dr. Pahde,

I am an MSc student working on shortcut mitigation in medical imaging, and I am reproducing your RR-ClArC result on
ISIC2019 with the controlled microscope artifact (Table 1, ResNet50: biased-test accuracy 0.76), using your public code
(github.com/frederikpahde/medical-ai-safety, commit 4990898). Thank you for releasing it; it made the replication
possible.

Our Vanilla model reproduces your numbers closely (clean 0.84, biased 0.31; your Table 1: 0.87 and 0.28). For RR-ClArC,
following the paper's stated settings (λ = 5·10⁹ from Table 12, learning rate = training learning rate / 10 = 1e-4 as on
p. 31, 10 epochs, fine-tuning from the Vanilla weights), we reach a biased-test accuracy of 0.52 (best of the three
layers in your config generator, chosen on validation), compared with 0.76 in the paper.

To make sure we have not missed a setting, could you share the exact configuration behind the ISIC2019 RR-ClArC result
in Table 1? In particular:

1. λ, and the layer the CAV and the RR-ClArC loss were applied to (we tried identity_1, identity_2 and last_conv);
2. the learning rate and the number of fine-tuning epochs;
3. the CAV settings (cav_mode, direction_mode, criterion, cav_scope) and the split used to compute the CAV;
4. how the Vanilla checkpoint was loaded for fine-tuning: in the released code, `get_optimizer` also loads the
   optimizer state from `ckpt_path`, which sets the learning rate to the decayed value of the Vanilla run (1e-5 after
   the 150/250 schedule) instead of the configured one. Was the fine-tune in the paper run with that optimizer state, or
   from the model weights only?
5. which timm weights the Table 1 ResNet50 (Vanilla) started from: `resnet50d.a1_in1k`, as loaded in
   `models/resnet_timm.py`, or `resnet50d.a3_in1k`, as stated on p. 22 (A.6)?

If the configuration files used for the paper are available, those would answer all of this at once. We will of course
report your answer, and cite your paper and code.

Thank you very much for your time.

Kind regards,
[Name]
[MSc programme, university]
