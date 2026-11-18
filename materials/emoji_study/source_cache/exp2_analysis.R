rm(list = ls(all = TRUE))

library(plyr)
library(ggplot2)
library(sciplot)
library(lme4)

####################################
# Demographic data
####################################

header <- names(read.csv("demographic.csv", header = TRUE, nrows = 1, skip = 0))
demographic <- read.csv("demographic.csv", header = FALSE, col.names = header, skip = 1)
demographic <- demographic[2:nrow(demographic),]

length.unique <- function(x){length(unique(x))}

# Confirm number of participants :

length.unique(demographic$Participant.Private.ID[demographic$Question.Key=="English-status"&demographic$Response=="Yes"])

# 60 native speaker participants

####################################
# Test data
####################################

header <- names(read.csv("protext.csv", header = TRUE, nrows = 1, skip = 0))
data <- read.csv("protext.csv", header = FALSE, col.names = header, skip = 1)
data <- data[2:nrow(data),]

data <- subset(data, Zone.Type=="response_slider_endValue")
data <- subset(data, Response != "")

is.numeric(data$Response)
data$Response <- as.numeric(data$Response)
is.numeric(data$Response)

# Check data :

length.unique(data$Emoji) # 9 emojis tested
length.unique(data$Environment) # 5 linguistic environments tested

# Rename factors and levels :

data$Condition <- ordered(data$Condition, levels=c("Target","Control"), labels=c("Target","Control"))
data$Environment <- ordered(data$Environment, levels=c("Unembedded","Question","Might","Negation","None"), labels=c("Unembedded","Question","Might","Negation","None"))

####################################
# Plot data
####################################

# Plot by environment :

dfc <- ddply(data, c("Participant.Private.ID","Condition","Environment"),
             function(data)c(Response=mean(data$Response)))
dfc1 <- ddply(dfc, c("Condition","Environment"),
              function(data)c(MEAN=mean(data$Response), SE=se(data$Response)))

boxplot.data <- aggregate(Response~Participant.Private.ID+Condition+Environment, FUN=mean, data=data)

plot <- ggplot(dfc1, aes(x=Environment, y=MEAN, fill=factor(Condition))) + 
  ggtitle("Endorsement of target and control inferences") + 
  xlab("Environment") + ylab("% endorsement") + 
  geom_bar(width=.9, position="dodge", stat="identity", alpha=.75) +
  scale_y_continuous(limits=c(NA,NA), breaks=c(0,25,50,75,100)) +
  theme_bw() + scale_fill_manual(name="Condition", 
                                 breaks=c("Target","Control"), labels=c("Target","Control"),
                                 values=c("orangered","royalblue3")) +
  theme(plot.title = element_text(hjust = 0.5)) +
  geom_errorbar(aes(ymin=MEAN-SE, ymax=MEAN+SE), width=.5, position=position_dodge(.9), col="black")
p <- 
  plot +
  geom_dotplot(
    data = boxplot.data, 
    aes(y=Response),
    col="black", alpha=.5, 
    binaxis='y', stackdir='center', binwidth = 1,
    position = "dodge"
  ) 
p
quartz(width=7,height=3)
print(p)

####################################
# Linear regression models
####################################

# Test for effect of condition (target vs. control), collapsing across environments) :

data$ConditionC <- ifelse(data$Condition=="Target", -.5, .5)
data.lm = lmer(Response~ConditionC+(1+ConditionC|Participant.Private.ID)+(1|Emoji), REML=FALSE, data=data)
data.lm_ConditionC = update(data.lm, .~.-ConditionC)
anova(data.lm, data.lm_ConditionC) # X^2(1)=123, p<.001 ***

# In more detail, test for effect of condition in each environment :

# UNEMBEDDED environment :

unem <- subset(data, Environment=="Unembedded")
unem$ConditionC <- ifelse(unem$Condition=="Target", -.5, .5)
unem.lm = lmer(Response~ConditionC+(1+ConditionC|Participant.Private.ID)+(1|Emoji), REML=FALSE, data=unem)
unem.lm_ConditionC = update(unem.lm, .~.-ConditionC)
anova(unem.lm, unem.lm_ConditionC) # X^2(1)=89, p<.001 ***

# QUESTION environment :

q <- subset(data, Environment=="Question")
q$ConditionC <- ifelse(q$Condition=="Target", -.5, .5)
q.lm = lmer(Response~ConditionC+(1+ConditionC|Participant.Private.ID)+(1|Emoji), REML=FALSE, data=q)
q.lm_ConditionC = update(q.lm, .~.-ConditionC)
anova(q.lm, q.lm_ConditionC) # X^2(1)=123, p<.001 ***

# MIGHT environment :

might <- subset(data, Environment=="Might")
might$ConditionC <- ifelse(might$Condition=="Target", -.5, .5)
might.lm = lmer(Response~ConditionC+(1+ConditionC|Participant.Private.ID)+(1|Emoji), REML=FALSE, data=might)
might.lm_ConditionC = update(might.lm, .~.-ConditionC)
anova(might.lm, might.lm_ConditionC) # X^2(1)=113, p<.001 ***

# NEGATION environment :

neg <- subset(data, Environment=="Negation")
neg$ConditionC <- ifelse(neg$Condition=="Target", -.5, .5)
neg.lm = lmer(Response~ConditionC+(1+ConditionC|Participant.Private.ID)+(1|Emoji), REML=FALSE, data=neg)
neg.lm_ConditionC = update(neg.lm, .~.-ConditionC)
anova(neg.lm, neg.lm_ConditionC) # X^2(1)=132, p<.001 *** 

# NONE environment :

none <- subset(data, Environment=="None")
none$ConditionC <- ifelse(none$Condition=="Target", -.5, .5)
none.lm = lmer(Response~ConditionC+(1+ConditionC|Participant.Private.ID)+(1|Emoji), REML=FALSE, data=none)
none.lm_ConditionC = update(none.lm, .~.-ConditionC)
anova(none.lm, none.lm_ConditionC) # X^2(1)=125, p<.001 ***