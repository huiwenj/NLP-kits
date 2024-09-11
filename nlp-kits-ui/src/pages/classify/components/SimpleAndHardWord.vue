<script setup>
import VChart from "vue-echarts";
import { wordPredict, wordTrain } from "@/api/classify";
import { use } from "echarts/core";
import {
  GridComponent,
  LegendComponent,
  TitleComponent,
  TooltipComponent,
} from "echarts/components";
import { LineChart } from "echarts/charts";
import { CanvasRenderer } from "echarts/renderers";
import { UniversalTransition } from "echarts/features";
import ContentCard from "@/components/ContentCard.vue";

const isTrain = ref(false);
const showSuccess = ref(false);

const text = ref("");
const predictVal = ref(-1);

const select = ref([0]);

const trainLosses = ref([]);
const valLosses = ref([]);

const emits = defineEmits(["update"]);

watchEffect(() => {
  const val = text.value;
  predictVal.value = -1;
});

const handleToTrain = async () => {
  isTrain.value = true;
  emits("update", { loading: isTrain.value });
  try {
    const { data } = await wordTrain();
    const { train_losses, val_losses } = data;
    trainLosses.value = train_losses;
    valLosses.value = val_losses;
    showSuccess.value = true;
  } finally {
    isTrain.value = false;
    emits("update", { loading: isTrain.value });
  }
};

const handlePredict = async () => {
  isTrain.value = true;
  emits("update", { loading: isTrain.value });
  try {
    const { data } = await wordPredict(text.value);
    predictVal.value = data;
  } finally {
    isTrain.value = false;
    emits("update", { loading: isTrain.value });
  }
};

use([
  TitleComponent,
  GridComponent,
  LegendComponent,
  LineChart,
  CanvasRenderer,
  UniversalTransition,
  TooltipComponent,
]);

const options = computed(() => {
  return {
    title: [
      {
        left: "center",
        text: "Training and Validation Loss",
      },
    ],
    legend: {
      data: ["Training Loss", "Validation Loss"],
      top: "top",
      right: "right",
    },
    xAxis: {
      type: "category",
      name: "Epoch",
      data: Array.from({ length: trainLosses.value.length }, (_, i) => i),
    },
    yAxis: {
      type: "value",
      name: "Loss",
    },
    tooltip: {
      trigger: "axis",
      axisPointer: {
        type: "cross",
        label: {
          backgroundColor: "#6a7985",
        },
      },
    },
    series: [
      {
        name: "Training Loss",
        data: trainLosses.value,
        type: "line",
        showSymbol: false,
        smooth: true,
        lineStyle: {
          width: 3,
        },
      },
      {
        name: "Validation Loss",
        data: valLosses.value,
        type: "line",
        showSymbol: false,
        smooth: true,
        lineStyle: {
          width: 3,
          lineStyle: {
            width: 3,
          },
        },
      },
    ],
  };
});
</script>

<template>
  <content-card v-bind="$attrs">
    <h2 class="select-none font-weight-medium">What is this model?</h2>
    <p class="text-indigo mt-2">
      <strong>Text classifier</strong>: distinguish between words that are
      simple and words that are complex by using RNN
    </p>

    <v-btn
      :loading="isTrain"
      class="mt-10"
      variant="tonal"
      color="purple"
      @click="handleToTrain"
    >
      Train
    </v-btn>

    <v-expansion-panels v-model="select" class="mt-3" v-if="showSuccess">
      <v-expansion-panel rounded elevation="0">
        <v-expansion-panel-title class="px-1 select-none font-weight-medium">
          Training Result
        </v-expansion-panel-title>
        <v-expansion-panel-text>
          <v-chart
            class="chart"
            style="height: 30vh"
            :option="options"
            autoresize
          />
        </v-expansion-panel-text>
      </v-expansion-panel>
    </v-expansion-panels>
  </content-card>

  <content-card class="mt-3">
    <h2 class="select-none font-weight-medium">Start to classifier</h2>
    <p class="text-indigo mt-2">
      Input a word, and the model will tell you whether it is a simple word or a
      hard word
    </p>

    <v-text-field
      class="mt-3"
      color="#8381C5"
      variant="outlined"
      max-width="400"
      clearable
      clear-icon="mdi-close"
      label="Word"
      placeholder="Input a word"
      outlined
      :disabled="isTrain"
      v-model="text"
    ></v-text-field>

    <v-btn
      :loading="isTrain"
      variant="tonal"
      color="purple"
      @click="handlePredict"
      :disabled="text === null || text === ''"
    >
      Predict
    </v-btn>

    <p class="mt-2" v-if="predictVal !== -1">
      Word <strong>{{ text }}</strong> is: a
      <strong>{{ predictVal === 0 ? "simple" : "hard" }}</strong> word
    </p>
  </content-card>
</template>

<style scoped></style>
