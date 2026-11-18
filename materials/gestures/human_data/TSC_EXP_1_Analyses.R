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

header <- names(read.csv("TSC_EXP_1_Data.csv", header = TRUE, nrows = 1, skip = 0))
data <- read.csv("TSC_EXP_1_Data.csv", header = FALSE, col.names = header, skip = 1)
data <- data[3:nrow(data),]

colnames(data) <- gsub("ResponseId", "SubjID", colnames(data))
data_cols <- grep("Test|Subj", colnames(data))
data <- subset(data, select=data_cols)
time_cols <- grep("time", colnames(data))
data <- subset(data, select=-time_cols)
colnames(data) <- gsub("_Test_1", "", colnames(data))

data <- melt(data, id.vars = c("SubjID","SubjYOB","SubjLanguage","SubjGender"), variable.name = "single_name", value.name = "value", na.rm=TRUE)
data <- subset(data, value != "")

design <- read.csv("TSC_EXP_1_Design.csv", header = TRUE)
df <- merge(design, data, by="single_name")

length.unique <- function(x){length(unique(x))}
length.unique(df$SubjID) # 103 participants 

xtabs(~df$SubjLanguage) # Participant languages


####################################
# Combined plot
####################################

df$value <- as.numeric(df$value)

si <- subset(df, Condition=="Scalar Implicature")

si$Environment <- ifelse(si$Group=="SI1a"|si$Group=="SI2a"|si$Group=="SI3a"|si$Group=="SI4a","Positive","Negative")
si$Environment <- factor(si$Environment, levels=c("Positive","Negative"), labels=c("Positive","Negative"))

si$Gesture <- ifelse(si$Environment=="Positive"&si$Gesture.Type=="X","Target",
                     ifelse(si$Environment=="Negative"&si$Gesture.Type=="not(X*)","Target","Control"))
si$Gesture <- factor(si$Gesture, levels=c("Target","Control"), labels=c("Target","Control"))

si$InferenceType <- ifelse(si$Inference.Type=="X-not-X*","Target","Baseline")
si$InferenceType <- factor(si$InferenceType, levels=c("Target","Baseline"), labels=c("Target","Baseline"))

pres <- subset(df, Condition=="Presupposition") 

pres$Environment <- ifelse(pres$Gesture.Type=="will-pq?","Question","None")
pres$Environment <- factor(pres$Environment, levels=c("Question","None"), labels=c("Question","None"))

pres$Gesture <- paste("Target")

pres$InferenceType <- ifelse(pres$Inference.Type=="p"|pres$Inference.Type=="everybody-p","Target","Baseline")
pres$InferenceType <- factor(pres$InferenceType, levels=c("Target","Baseline"), labels=c("Target","Baseline"))

supp <- subset(df, Condition=="Supplement")

supp$Environment <- paste("Unembedded")

supp$Gesture <- ifelse(supp$Gesture.Type=="if-A(supp)-then-C","Target","Control")
supp$Gesture <- factor(supp$Gesture, levels=c("Target","Control"), labels=c("Target","Control"))

supp$InferenceType <- ifelse(supp$Inference.Type=="if-A-then-supp","Target","Baseline")
supp$InferenceType <- factor(supp$InferenceType, levels=c("Target","Baseline"), labels=c("Target","Baseline"))

hmg <- subset(df, Condition=="Homogeneity")

hmg$Environment <- ifelse(hmg$Gesture.Type=="the-Xs","Positive","Negative")
hmg$Environment <- factor(hmg$Environment, levels=c("Positive","Negative"), labels=c("Positive","Negative"))

hmg$Gesture <- paste("Target")

hmg$Inference <- ifelse(hmg$Inference.Type=="no-Xs","None",
                        ifelse(hmg$Inference.Type=="all-Xs","All","Some but not all"))
hmg$InferenceType <- ifelse((hmg$Inference=="All"|hmg$Inference=="None"), "Target","Baseline")
hmg$InferenceType <- factor(hmg$InferenceType, levels=c("Target","Baseline"), labels=c("Target","Baseline"))

# Plot :

sip <- rbind(si,pres)
sips <- rbind(sip,supp)
sipsh <- rbind(sips,hmg)

sipsh$Condition <- ordered(sipsh$Condition, levels=c("Scalar Implicature","Presupposition","Supplement","Homogeneity"), labels=c("Scalar Implicature","Presupposition","Supplement","Homogeneity"))

all_dfc <- ddply(sipsh, c("SubjID","Condition","Environment","Gesture","InferenceType"),
                 function(sipsh)c(value=mean(sipsh$value)))
all_dfc1 <- ddply(all_dfc, c("Condition","Environment","Gesture","InferenceType"),
                  function(sipsh)c(MEAN=mean(sipsh$value), SE=se(sipsh$value)))

all.boxplot.data <- aggregate(value~SubjID+Condition+Environment+Gesture+InferenceType, FUN=mean, data=sipsh)

plot_all <- ggplot(all_dfc1, aes(x=Gesture, y=MEAN, fill=factor(InferenceType))) +
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

# Scalar implicatures (positive) 

dsi <- subset(si,Group=="SI1a"|Group=="SI2a"|Group=="SI3a"|Group=="SI4a")

dsi$GestureC <- ifelse(dsi$Gesture=="Target",-.5,.5)
dsi$InferenceC <- ifelse(dsi$InferenceType=="Target",-.5,.5)

dsi.lm = lmer(value~GestureC*InferenceC+(1+GestureC*InferenceC|SubjID)+(1|Group), REML=FALSE, data=dsi) 
# NB: The version of this model without the interaction failed to converge, leading us to simplify the random effects structure in the following way:
dsi.lm = lmer(value~GestureC*InferenceC+(1+GestureC*InferenceC|SubjID), REML=FALSE, data=dsi) 
# NB: The version of this model without the interaction was nearly unidentifiable, leading us to further simplify the random effects structure:
dsi.lm = lmer(value~GestureC*InferenceC+(1+GestureC+InferenceC|SubjID), REML=FALSE, data=dsi) 
dsi_Gesture = update(dsi.lm, .~.-GestureC)
dsi_Inference = update(dsi.lm, .~.-InferenceC)
dsi_GI = update(dsi.lm, .~.-GestureC:InferenceC)

anova(dsi.lm, dsi_Gesture) # No effect of Gestural Premise (X^2(1)=2.3, p=.13)
anova(dsi.lm, dsi_Inference) # Significant effect of Inference Type (X^2(1)=13, p<.001)
anova(dsi.lm, dsi_GI) # Significant interaction between Gestural Premise x Inference Type (X^2(1)=3089, p<.001) 

# Scalar implicatures (negative)

isi <- subset(si, Group=="SI1b"|Group=="SI2b"|Group=="SI3b"|Group=="SI4b")

isi$GestureC <- ifelse(isi$Gesture=="Target",-.5,.5)
isi$InferenceC <- ifelse(isi$InferenceType=="Target",-.5,.5)

isi.lm = lmer(value~GestureC*InferenceC+(1+GestureC*InferenceC|SubjID)+(1|Group), REML=FALSE, data=isi)
isi_Inference = update(isi.lm, .~.-InferenceC)
isi_Gesture = update(isi.lm, .~.-GestureC)
isi_IG = update(isi.lm, .~.-InferenceC:GestureC)

anova(isi.lm, isi_Gesture) # Significant effect of Gestural Premise (X^2(1)=12, p<.001)
anova(isi.lm, isi_Inference) # Marginal effect of Inference Type (X^2(1)=3.0, p=.086)
anova(isi.lm, isi_IG) # Significant interaction between Gestural Premise x Inference Type (X^2(1)=55, p<.001) 

#===================================
# Presuppositions
#===================================

# Presuppositions (questions)

q <- subset(pres, Gesture.Type=="will-pq?")

q$InferenceC <- ifelse(q$InferenceType=="Target",-.5,.5)

q.lm = lmer(value~InferenceC+(1+InferenceC|SubjID)+(1|Group), REML=FALSE, data=q)
q_Inference = update(q.lm, .~.-InferenceC)

anova(q.lm, q_Inference) # Significant effect of Inference Type (X^2(1)=230, p<.001)

# Presuppositions ("none")

none <- subset(pres, Gesture.Type=="nobody-pq")

none$InferenceC <- ifelse(none$InferenceType=="Target",-.5,.5)

none.lm = lmer(value~InferenceC+(1+InferenceC|SubjID)+(1|Group), REML=FALSE, data=none)
none_Inference = update(none.lm, .~.-InferenceC)

anova(none.lm, none_Inference) # Significant effect of Inference Type (X^2(1)=14, p<.001)

#===================================
# Supplements
#===================================

supp$InferenceC <- ifelse(supp$InferenceType=="Target",-.5,.5)
supp$GestureC <- ifelse(supp$Gesture=="Target",-.5,.5)

supp.lm = lmer(value~GestureC*InferenceC+(1+GestureC+InferenceC|SubjID)+(1|Group), REML=FALSE, data=supp)
supp_Gesture = update(supp.lm, .~.-GestureC)
supp_Inference = update(supp.lm, .~.-InferenceC)
# NB: The previous two models fail to converge, but they pertain to the main effects, which are not relevant to our inquiry. The models have been included here for the sake of completeness, but are not interpreted or reported in the paper.
supp_GI = update(supp.lm, .~.-GestureC:InferenceC)

anova(supp.lm, supp_Gesture) # No effect of Gestural Premise (chi^2(1)=1.5, p=.22)
anova(supp.lm, supp_Inference) # Significant effect of Inference type (chi^2(1)=30, p<.001)
anova(supp.lm, supp_GI) # Significant interaction between Gestural Premise x Inference Type (X^2(1)=27, p<.001)

#===================================
# Homogeneity
#===================================

# Homogeneity (positive)

positive <- subset(hmg, Environment=="Positive")

positive$InferenceTypeC <- ifelse(positive$InferenceType=="Target",-.5,.5)

positive.lm = lmer(value~InferenceTypeC+(1+InferenceTypeC|SubjID)+(1|Group), REML=FALSE, data=positive)
positive_I = update(positive.lm, .~.-InferenceTypeC)

anova(positive.lm, positive_I) # Significant effect of Inference type (X^2(1)=84, p<.001)

# Homogeneity (negative)

negative <- subset(hmg, Environment=="Negative")

negative$InferenceTypeC <- ifelse(negative$InferenceType=="Target",-.5,.5)

negative.lm = lmer(value~InferenceTypeC+(1+InferenceTypeC|SubjID), REML=FALSE, data=negative)
negative_I = update(negative.lm, .~.-InferenceTypeC)

anova(negative.lm, negative_I) # Significant effect of Inference type (X^2(1)=132, p<.001)

#===================================
# Iconicity controls
#===================================

iconicity <- subset(df, Condition=="Iconicity") 

iconicity$Inference <- ifelse(iconicity$Inference.Type=="match","Match","Mismatch")

iconicity_dfc <- ddply(iconicity, c("SubjID","Gesture.Type","Inference"),
                       function(iconicity)c(value=mean(iconicity$value)))
iconicity_dfc2 <- ddply(iconicity_dfc, c("Gesture.Type","Inference"),
                        function(iconicity)c(MEAN=mean(iconicity$value), SE=se(iconicity$value)))
iconicity_dfc1 <- ddply(iconicity_dfc, c("Inference"),
                        function(iconicity)c(MEAN=mean(iconicity$value), SE=se(iconicity$value)))
iconicity_dfc1

# Inference      MEAN        SE
# 1     Match 73.728155 1.6127012
# 2  Mismatch  7.259709 0.7574277

# Model :

iconicity$InferenceC <- ifelse(iconicity$Inference=="Match",-.5,.5)

iconicity.lm = lmer(value~InferenceC+(1+InferenceC|SubjID)+(1|Group), REML=FALSE, data=iconicity)
iconicity_InferenceC = update(iconicity.lm, .~.-InferenceC)

anova(iconicity.lm, iconicity_InferenceC) # Significant effect of Match Type (match vs. mismatch) (X^2(1)=230, p<.001)