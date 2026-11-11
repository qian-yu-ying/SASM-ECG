# Upstream materials and attribution

PTB-XL: Wagner et al., Scientific Data 7, 154 (2020),
doi:10.1038/s41597-020-0495-6. PhysioNet PTB-XL 1.0.3 specifies CC BY 4.0.
https://physionet.org/content/ptb-xl/1.0.3/

Ningbo: Zheng, Guo and Chu, PhysioNet, A large scale 12-lead electrocardiogram
database for arrhythmia study, v1.0.0, doi:10.13026/wgex-er52.
The PhysioNet release specifies CC BY 4.0.
https://physionet.org/content/ecg-arrhythmia/1.0.0/

CPSC2018: Liu et al., Journal of Medical Imaging and Health Informatics 8(7),
1368–1373 (2018), doi:10.1166/jmihi.2018.2442. The PhysioNet Challenge 2020
distribution, which includes CPSC2018, specifies CC BY 4.0.
https://physionet.org/content/challenge-2020/1.0.2/

SPH: Liu et al., Scientific Data 9, 272 (2022),
doi:10.1038/s41597-022-01403-5. Figshare collection 5779802 includes the ECG
records (article 17912444); that article specifies CC0.
https://doi.org/10.6084/m9.figshare.c.5779802.v1
https://doi.org/10.6084/m9.figshare.17912444.v1

ECGFounder: Li et al., NEJM AI 2(7) (2025), doi:10.1056/AIoa2401033.
Official repository: https://github.com/NickLJLee/ECGFounder
Official checkpoint: https://huggingface.co/PKUDigitalHealth/ECGFounder
Repository and checkpoint metadata specify MIT. The published affine native
projection is included as a derived asset; no complete encoder is redistributed.

MERL: Liu et al., Zero-Shot ECG Classification with Multimodal Learning and
Test-time Clinical Knowledge Enhancement, ICML 2024, PMLR 235:31949–31963.
Official repository: https://github.com/cheliu-computation/MERL
The repository specifies MIT. No complete encoder is redistributed.

Features in this package were computed using the study's fixed preprocessing
and checkpoints. They are transformed, filtered and reindexed derivatives,
not original signals. Source classifiers were fitted on PTB-XL; upstream
authors do not endorse the present method or release. Retain attribution when
redistributing dataset derivatives. NumPy, SciPy, scikit-learn and threadpoolctl
are runtime dependencies and are not vendored in the release.
