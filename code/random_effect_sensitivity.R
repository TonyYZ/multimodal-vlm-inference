# Optional helpers for reproduce_original_human_tests.R --sensitivity.
# Candidate structures are fixed in advance and never selected by a diagnostic p-value.
sensitivity_formulas <- function(template, term) {
  story_present <- any(vapply(findbars(template), function(bar) {
    identical(as.character(bar[[3]]), "Story")
  }, logical(1)))
  fixed <- if (grepl(":", term)) "InferenceC * GestureC" else term
  list(
    lyn_final=template,
    lyn_with_story=if (story_present) template else update.formula(template, . ~ . + (1 | Story)),
    uncorrelated_slopes=as.formula(paste(
      "discrepancy ~", fixed, "+ (1 +", fixed, "|| SubjID) + (1 | Story)"
    )),
    intercepts_only=as.formula(paste("discrepancy ~", fixed, "+ (1 | SubjID) + (1 | Story)"))
  )
}

fit_converged <- function(model) {
  messages <- model@optinfo$conv$lme4$messages
  messages <- messages[!grepl("^boundary \\(singular\\)", messages)]
  all(model@optinfo$conv$opt == 0) && length(messages) == 0
}

sensitivity_tests <- function(template, data, term, preparation_error=NULL) {
  specifications <- sensitivity_formulas(template, term)
  cache <- list()
  rows <- lapply(names(specifications), function(variant) {
    specification <- specifications[[variant]]
    signature <- paste(deparse(specification), collapse=" ")
    if (!is.null(cache[[signature]])) {
      return(cbind(variant=variant, cache[[signature]]))
    }
    # update() must see this fit's data, not another diagnostic's data.
    environment(specification) <- environment()
    result <- tryCatch({
      if (!is.null(preparation_error)) stop(preparation_error)
      full <- lmer(specification, REML=FALSE, data=data,
                   control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=200000)))
      reduced <- update(full, as.formula(paste(". ~ . -", term)))
      comparison <- anova(full, reduced)
      comparison <- comparison[!is.na(comparison$Chisq), , drop=FALSE]
      stopifnot(nrow(comparison) == 1)
      residual <- residuals(full) / sigma(full)
      fitted <- fitted(full)
      messages <- unique(c(full@optinfo$conv$lme4$messages, reduced@optinfo$conv$lme4$messages))
      data.frame(
        coefficient_term=term, estimate=unname(fixef(full)[[term]]),
        std_error=unname(sqrt(diag(vcov(full)))[[term]]),
        chi_square=comparison$Chisq, df=comparison$Df, p_value=comparison[["Pr(>Chisq)"]],
        n_observations=nobs(full), singular_fit=isSingular(full) || isSingular(reduced),
        convergence_message=paste(messages, collapse="; "), analysis_status="ok",
        candidate_formula=signature, full_converged=fit_converged(full),
        reduced_converged=fit_converged(reduced), full_singular=isSingular(full),
        reduced_singular=isSingular(reduced), aic_full=AIC(full),
        residual_rmse=sqrt(mean(residuals(full)^2)),
        residual_outlier_fraction=mean(abs(residual) > 3),
        absolute_residual_fitted_rho=if (sd(fitted) > 0 && sd(abs(residual)) > 0) {
          cor(abs(residual), fitted, method="spearman")
        } else NA_real_
      )
    }, error=function(error) {
      data.frame(
        coefficient_term=term, estimate=NA_real_, std_error=NA_real_, chi_square=NA_real_,
        df=NA_real_, p_value=NA_real_, n_observations=0L, singular_fit=NA,
        convergence_message="", analysis_status=conditionMessage(error),
        candidate_formula=signature, full_converged=FALSE, reduced_converged=FALSE,
        full_singular=NA, reduced_singular=NA, aic_full=NA_real_, residual_rmse=NA_real_,
        residual_outlier_fraction=NA_real_, absolute_residual_fitted_rho=NA_real_
      )
    })
    cache[[signature]] <<- result
    cbind(variant=variant, result)
  })
  do.call(rbind, rows)
}
