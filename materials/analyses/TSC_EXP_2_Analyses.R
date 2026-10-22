rm(list = ls(all = TRUE))

library(rmarkdown)
library(plyr)
library(dplyr)
library(ggplot2)
library(reshape2)
library(sciplot)
library(lme4)


####################################
# Prepare Qualtrics data
####################################

header <- names(read.csv("TSC_EXP_2_Data.csv", header = TRUE, nrows = 1, skip = 0))
data <- read.csv("TSC_EXP_2_Data.csv", header = FALSE, col.names = header, skip = 1)
data <- data[3:nrow(data),]

colnames(data) <- gsub("ResponseId", "SubjID", colnames(data))
data_cols <- grep("Test|Subj|Duration", colnames(data))
data <- subset(data, select=data_cols)
time_cols <- grep("time", colnames(data))
data <- subset(data, select=-time_cols)
colnames(data) <- gsub("_Test_2", "", colnames(data))
colnames(data) <- gsub("Duration..in.seconds.", "Duration", colnames(data))

data <- melt(data, id.vars = c("SubjID","SubjYOB","SubjLanguage","SubjGender","Duration"), variable.name = "qualtrics", value.name = "value", na.rm=TRUE)
data <- subset(data, value != "")

design <- read.csv("TSC_EXP_2_Design.csv", header = TRUE)
df <- merge(design, data, by="qualtrics")

length.unique <- function(x){length(unique(x))}
length.unique(df$SubjID) # 99 participants

xtabs(~df$SubjLanguage) # Participant languages


####################################
# Combined plot
####################################

df$value <- as.numeric(as.character(df$value))

df$Condition <- ordered(df$Condition, levels=c("Implicature","Presupposition","Supplement","Homogeneity"), labels=c("Scalar Implicature","Presupposition","Supplement","Homogeneity"))

df$Environment <- ordered(df$Environment, levels=c("Positive","Negative","Question","None","Unembedded"))

df$GesturalPremise <- ordered(df$GesturalPremise, levels=c("Target","Control"))
df$InferenceType <- ifelse(df$InferenceType=="Target","Target","Baseline")
df$InferenceType <- ordered(df$InferenceType, levels=c("Target","Baseline"))

df_dfc <- ddply(df, c("SubjID","Condition","Environment","GesturalPremise","InferenceType"),
                 function(df)c(value=mean(df$value)))
df_dfc1 <- ddply(df_dfc, c("Condition","Environment","GesturalPremise","InferenceType"),
                  function(df)c(MEAN=mean(df$value), SE=se(df$value)))

all.boxplot.data <- aggregate(value~SubjID+Condition+Environment+GesturalPremise+InferenceType, FUN=mean, data=df)

plot_all <- ggplot(df_dfc1, aes(x=GesturalPremise, y=MEAN, fill=factor(InferenceType))) +
  ggtitle("") +
  xlab("Premises") + ylab("Mean endorsement") +
  geom_bar(width=.9, position="dodge", stat="identity", alpha = .5) +
  scale_y_continuous(limits=c(NA,NA), breaks=c(0,25,50,75,100), labels = function(x) paste0(x,"%")) +
  theme_bw() +
  theme(plot.title = element_text(hjust=0.5), text = element_text(size=14)) +
  scale_fill_manual(name="Inferences",
                    breaks=c("Target","Baseline"), labels=c("Target","Baseline"),
                    values=c("seagreen2","darkgrey")) +
  facet_grid(.~Condition*Environment, scales="free") +
  geom_errorbar(aes(ymin=MEAN-SE, ymax=MEAN+SE), width=.5, position=position_dodge(.9), col="black") +
  scale_colour_manual(guide=FALSE, values=c("seagreen2","darkgrey"))
p <- 
  plot_all +
  geom_dotplot(
    data = all.boxplot.data, 
    aes(y=value),
    col="black", alpha=.25, 
    binaxis='y', stackdir='center', binwidth = 1,
    position = "dodge"
  ) 
p
quartz(width=11.75,height=3.25)
print(p)


####################################
# Models
####################################

#===================================
# Scalar implicatures
#===================================

si <- subset(df, Condition=="Scalar Implicature")

si$Environment <- factor(si$Environment, levels=c("Positive","Negative"), labels=c("Positive","Negative"))
si$GesturalPremise <- factor(si$GesturalPremise, levels=c("Target","Control"), labels=c("Target","Control"))
si$InferenceType <- factor(si$InferenceType, levels=c("Target","Baseline"), labels=c("Target","Baseline"))

# Scalar implicatures (positive)

dsi <- subset(si,Environment=="Positive")

dsi$GestureC <- ifelse(dsi$GesturalPremise=="Target",-.5,.5)
dsi$InferenceC <- ifelse(dsi$InferenceType=="Target",-.5,.5)

dsi.lm = lmer(value~InferenceC*GestureC+(1+InferenceC*GestureC|SubjID)+(1|Story), REML=FALSE, data=dsi) 
# NB: This model failed to converge, leading us to simplify the random effects structure in the following way:
dsi.lm = lmer(value~InferenceC*GestureC+(1+InferenceC*GestureC|SubjID), REML=FALSE, data=dsi)
# NB: This model failed to converge, leading us to simplify the random effects structure in the following way:
dsi.lm = lmer(value~InferenceC*GestureC+(1+InferenceC+GestureC|SubjID), REML=FALSE, data=dsi)
dsi_Gesture = update(dsi.lm, .~.-GestureC)
dsi_Inference = update(dsi.lm, .~.-InferenceC)
dsi_GestureInference = update(dsi.lm, .~.-GestureC:InferenceC)

anova(dsi.lm, dsi_Gesture) # Significant effect of Gesture Type (X^2(1)=5.0, p<.05) 
anova(dsi.lm, dsi_Inference) # Significant effect of Inference Type (X^2(1)=13, p<.001) 
anova(dsi.lm, dsi_GestureInference) # Significant interaction between Gesture Type x Inference Type (X^2(1)=1361, p<.001) 

# Scalar implicatures (negative)

isi <- subset(si, Environment=="Negative")

isi$GestureC <- ifelse(isi$GesturalPremise=="Target",-.5,.5)
isi$InferenceC <- ifelse(isi$InferenceType=="Target",-.5,.5)

isi.lm = lmer(value~InferenceC*GestureC+(1+InferenceC*GestureC|SubjID)+(1|Story), REML=FALSE, data=isi)
isi_Inference = update(isi.lm, .~.-InferenceC)
isi_Gesture = update(isi.lm, .~.-GestureC)
isi_InferenceGesture = update(isi.lm, .~.-InferenceC:GestureC)

anova(isi.lm, isi_Inference) # Significant effect of Inference Type (X^2(1)=5.4, p<.05) 
anova(isi.lm, isi_Gesture) # Marginal effect of Gesture Type (X^2(1)=3.0, p=.08) 
anova(isi.lm, isi_InferenceGesture) # Significant interaction between Inference Type x Gesture Type (X^2(1)=60, p<.001) 

#===================================
# Presuppositions
#===================================

pres <- subset(df, Condition=="Presupposition") 

pres$Environment <- factor(pres$Environment, levels=c("Question","None"), labels=c("Question","None"))
pres$GesturalPremise <- factor(pres$GesturalPremise, levels=c("Target","Control"), labels=c("Target","Control"))
pres$InferenceType <- factor(pres$InferenceType, levels=c("Target","Baseline"), labels=c("Target","Baseline"))

# Presuppositions (questions)

q <- subset(pres, Environment=="Question")

q$InferenceC <- ifelse(q$InferenceType=="Target",-.5,.5)

q.lm = lmer(value~InferenceC+(1+InferenceC|SubjID)+(1|Story), REML=FALSE, data=q)
# NB: The version of this model without the Inference Type factor failed to converge, leading us to simplify the random effects structure in the following way:
q.lm = lmer(value~InferenceC+(1+InferenceC|SubjID), REML=FALSE, data=q)
# NB: The version of this model without the Inference Type factor failed to converge, leading us to further simplify the random effects structure:
q.lm = lmer(value~InferenceC+(1|SubjID), REML=FALSE, data=q)
q_Inference = update(q.lm, .~.-InferenceC)

anova(q.lm, q_Inference) # Significant effect of Inference Type (X^2(1)=49, p<.001) 

# Presuppositions ("none")

none <- subset(pres, Environment=="None")

none$InferenceC <- ifelse(none$InferenceType=="Target",-.5,.5)

none.lm = lmer(value~InferenceC+(1+InferenceC|SubjID)+(1|Story), REML=FALSE, data=none)
none_Inference = update(none.lm, .~.-InferenceC)

anova(none.lm, none_Inference) # Significant effect of Inference Type (X^2(1)=79, p<.001) 

#===================================
# Supplements
#===================================

supp <- subset(df, Condition=="Supplement")

supp$GesturalPremise <- factor(supp$GesturalPremise, levels=c("Target","Control"), labels=c("Target","Control"))
supp$InferenceType <- factor(supp$InferenceType, levels=c("Target","Baseline"), labels=c("Target","Baseline"))

supp$InferenceC <- ifelse(supp$InferenceType=="Target",-.5,.5)
supp$GestureC <- ifelse(supp$GesturalPremise=="Target",-.5,.5)

supp.lm = lmer(value~InferenceC*GestureC+(1+InferenceC*GestureC|SubjID)+(1|Story), REML=FALSE, data=supp)
supp_Gesture = update(supp.lm, .~.-GestureC)
supp_Inference = update(supp.lm, .~.-InferenceC)
supp_GestureInference = update(supp.lm, .~.-GestureC:InferenceC)

anova(supp.lm, supp_Gesture) # No effect of Gesture Type (X^2(1)=.32, p=.57)
anova(supp.lm, supp_Inference) # Significant effect of Inference Type (X^2(1)=96, p<.001) 
anova(supp.lm, supp_GestureInference) # Marginal interaction between Gesture Type x Inference Type (X^2(1)=3.5, p=.06) 

#===================================
# Homogeneity
#===================================

hmg <- subset(df, Condition=="Homogeneity")

hmg$Environment <- factor(hmg$Environment, levels=c("Positive","Negative"), labels=c("Positive","Negative"))
hmg$InferenceType <- factor(hmg$InferenceType, levels=c("Target","Baseline"), labels=c("Target","Baseline"))

# Homogeneity (positive)

positive <- subset(hmg, Environment=="Positive")

positive$InferenceTypeC <- ifelse(positive$InferenceType=="Target",-.5,.5)

positive.lm = lmer(value~InferenceTypeC+(1+InferenceTypeC|SubjID)+(1|Story), REML=FALSE, data=positive)
positive_Inference = update(positive.lm, .~.-InferenceTypeC)

anova(positive.lm, positive_Inference) # Significant effect of Inference Type (X^2(1)=42, p<.001) 

# Homogeneity (negative)

negative <- subset(hmg, Environment=="Negative")

negative$InferenceTypeC <- ifelse(negative$InferenceType=="Target",-.5,.5)

negative.lm = lmer(value~InferenceTypeC+(1+InferenceTypeC|SubjID)+(1|Story), REML=FALSE, data=negative)
negative_Inference = update(negative.lm, .~.-InferenceTypeC)

anova(negative.lm, negative_Inference) # Significant effect of Inference Type (X^2(1)=87, p<.001) 