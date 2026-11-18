# Called by reproduce_original_human_tests.R after Lyn's unchanged human analyses.
.libPaths(c(file.path(root, "tmp", "R-library"), .libPaths()))
if (!requireNamespace("lmtest", quietly=TRUE)) stop("Install the R package lmtest before running VLM LRTs.")
rating_dir <- file.path(root, "results", study, "inference",
                        if (study == "gesture") "full_mask_note" else "ratings")
paths <- list.files(rating_dir, pattern="ratings\\.jsonl$", full.names=TRUE)
paths <- paths[!grepl("_presupposition_positive_", basename(paths))]
read_scores <- function(path) {
  con <- file(path, open="r", encoding="UTF-8")
  x <- tryCatch(stream_in(con, verbose=FALSE), finally=close(con))
  x$source_video <- basename(gsub("\\\\", "/", x$source_video))
  x$statement_type <- tolower(x$statement_type)
  text <- sub("^\\s*thought[ \\t]*\\r?\\n", "", x$response, ignore.case=TRUE, perl=TRUE)
  x$value <- strcapture("^\\s*([0-9]{1,3}(?:\\.[0-9]+)?)(?:\\s|$)", text,
                       proto=list(value=double()), perl=TRUE)$value
  good <- is.finite(x$value) & x$value >= 0 & x$value <= 100
  if ("error" %in% names(x)) good <- good & (is.na(x$error) | x$error == "")
  x <- x[good, ]
  x[!duplicated(x[c("model", "input_mode", "source_video", "statement_type")], fromLast=TRUE), ]
}
ratings <- do.call(rbind, lapply(paths, function(path) {
  x <- read_scores(path)
  x[c("model", "input_mode", "source_video", "statement_type", "value")]
}))
tests <- cells <- list()
for (i in seq_len(nrow(diagnostics))) {
  diagnostic <- diagnostics[i, ]
  h <- human[[diagnostic$dataset]]
  if (study == "gesture") {
    h$GesturalPremise <- h$Gesture
    h$Story <- regmatches(h$Group, regexpr("[0-9]+", h$Group))
  }
  keys <- c("Story", "GesturalPremise", "InferenceType")
  design <- unique(h[keys])
  if (study == "gesture") {
    # The human response file lacks Supplement2 control columns; the design does not.
    raw <- subset(human$design, Condition == diagnostic$condition)
    raw$Story <- regmatches(raw$Group, regexpr("[0-9]+", raw$Group))
    raw$Environment <- switch(diagnostic$condition,
      "Scalar Implicature"=ifelse(grepl("a$", raw$Group), "Positive", "Negative"),
      "Presupposition"=ifelse(raw$Gesture.Type == "will-pq?", "Question", "None"),
      "Homogeneity"=ifelse(raw$Gesture.Type == "the-Xs", "Positive", "Negative"), "Unembedded")
    raw$GesturalPremise <- ifelse(
      raw$Condition == "Scalar Implicature" & !((raw$Environment == "Positive" & raw$Gesture.Type == "X") |
        (raw$Environment == "Negative" & raw$Gesture.Type == "not(X*)")) |
      raw$Condition == "Supplement" & raw$Gesture.Type != "if-A(supp)-then-C", "Control", "Target")
    raw$InferenceType <- ifelse(raw$Inference.Type %in%
      c("X-not-X*", "p", "everybody-p", "all-Xs", "no-Xs", "if-A-then-supp"), "Target", "Baseline")
    design <- unique(subset(raw, Environment == diagnostic$environment)[keys])
  }
  design$source_video <- paste0(gsub(" ", "", diagnostic$condition), design$Story,
    "_", design$GesturalPremise, "Premise",
    if (diagnostic$environment == "Unembedded") "" else paste0("(", diagnostic$environment, ")"),
    if (study == "gesture") ".mp4" else ".m4v")
  design$statement_type <- tolower(as.character(design$InferenceType))
  human_means <- aggregate(h$value, h[keys], mean)
  names(human_means)[names(human_means) == "x"] <- "human_mean"
  design <- merge(design, human_means, by=keys, all.x=TRUE)
  three_way <- diagnostic$condition %in% c("Scalar Implicature", "Supplement")
  formula <- if (three_way) value ~ InferenceC * GestureC else value ~ InferenceC
  term <- if (three_way) "InferenceC:GestureC" else "InferenceC"
  for (group in split(ratings, interaction(ratings$model, ratings$input_mode, drop=TRUE))) {
    d <- merge(design, group, by=c("source_video", "statement_type"))
    d$InferenceC <- ifelse(d$InferenceType == "Target", -0.5, 0.5)
    d$GestureC <- ifelse(d$GesturalPremise == "Target", -0.5, 0.5)
    result <- data.frame(estimate=NA_real_, std_error=NA_real_, chi_square=NA_real_,
                         df=1, residual_df=NA_real_, p_value=NA_real_,
                         log_likelihood_full=NA_real_, log_likelihood_reduced=NA_real_,
                         ci_low=NA_real_, ci_high=NA_real_, analysis_status="incomplete cells")
    if (nrow(d) == nrow(design)) {
      fit <- lm(formula, data=d)
      reduced <- update(fit, as.formula(paste(". ~ . -", term)))
      result$estimate <- unname(coef(fit)[[term]])
      result$residual_df <- df.residual(fit)
      # Exact fits have no estimable residual uncertainty: retain coefficients, not p=0.
      if (df.residual(fit) > 0 && deviance(fit) > 1e-10 * max(1, sum(d$value^2))) {
        comparison <- lmtest::lrtest(reduced, fit)
        result$std_error <- coef(summary(fit))[term, "Std. Error"]
        result$chi_square <- comparison$Chisq[2]
        result$df <- comparison$Df[2]
        result$p_value <- comparison[["Pr(>Chisq)"]][2]
        result$log_likelihood_full <- as.numeric(logLik(fit, REML=FALSE))
        result$log_likelihood_reduced <- as.numeric(logLik(reduced, REML=FALSE))
        result[1, c("ci_low", "ci_high")] <- confint(fit, term)
        result$analysis_status <- "ok"
      } else result$analysis_status <- "zero residual variance; inference unavailable"
    }
    prefix <- data.frame(model=group$model[1], input_format=group$input_mode[1],
                         condition=diagnostic$condition, environment=diagnostic$environment)
    tests[[length(tests)+1]] <- cbind(prefix, coefficient_term=term, result,
      n_observations=nrow(d), n_stories=length(unique(d$Story)),
      n_cells_with_human_data=sum(!is.na(d$human_mean)),
      model_formula=paste(deparse(formula), collapse=" "),
      reduced_formula=paste(deparse(update.formula(formula, as.formula(paste(". ~ . -", term)))), collapse=" "),
      test_design="Gaussian fixed-effects ML likelihood-ratio test (lmtest::lrtest)")
    cells[[length(cells)+1]] <- cbind(prefix[rep(1, nrow(d)), ],
      d[setdiff(names(d), c("model", "input_mode"))])
  }
}
write.csv(do.call(rbind, tests), file.path(output_dir, "llm_fixed_effect_tests.csv"), row.names=FALSE)
write.csv(do.call(rbind, cells), file.path(output_dir, "llm_fixed_effect_cells.csv"), row.names=FALSE)
cat("Wrote", study, "VLM fixed-only models; no participant/Story random effects or replicated LLM subjects.\n")
