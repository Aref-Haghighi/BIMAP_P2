
# overlays:

IMG1_overlay.png: RAW vs DENOISED (both with scalebar)

IMG1_difference.png:Absolute difference (luma) + colorbar

IMG1_difference_signed.png:   Signed difference (denoised − raw)

IMG1_difference_background.png.  Abs-diff masked to background + metrics box

IMG1_overlay_diff.png:     Triptych: RAW | DENOISED | ABS-DIFF + metrics

IMG1_line_profile.png:   Edge line-profile and ΔFWHM

IMG1_radial_power.png:    Radial power spectrum (RAW vs DENOISED)



- IMG1_denoised_R.npy:                N2V cache for Red channel (raw units)
- IMG1_denoised_G.npy:                N2V cache for Green channel (raw units)
- IMG1_denoised_B.npy:                N2V cache for Blue channel (raw units)
- psnr_ssim_3ch.csv:                  Per-image/channel PSNR/SSIM + Δ-metrics
- summary_deltas.png:                 Boxplots of ΔBRISQUE, ΔNIQE, Δσ_bg, ΔCNR(gm), ΔFWHM
- summary_table_deltas.png:           Table with mean ± SD and N for the Δ-metrics

