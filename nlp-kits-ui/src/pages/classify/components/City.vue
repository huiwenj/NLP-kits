<script setup>
import VChart from "vue-echarts";
import { cityList, cityPredict, cityTrain } from "@/api/classify";
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

const city = ref(null);
const cities = ref([]);
const predictVal = ref(-1);

const select = ref([0]);

const trainLosses = ref([]);
const valLosses = ref([]);

const emits = defineEmits(["update"]);

const countries = [
  {
    id: "af",
    name: "Afghanistan",
  },
  {
    id: "cn",
    name: "China",
  },
  {
    id: "de",
    name: "Germany",
  },
  {
    id: "fi",
    name: "Finland",
  },
  {
    id: "fr",
    name: "France",
  },
  {
    id: "in",
    name: "India",
  },
  {
    id: "ir",
    name: "Iran",
  },
  {
    id: "pk",
    name: "Pakistan",
  },
  {
    id: "za",
    name: "South Africa",
  },
];

watchEffect(() => {
  const val = city.value;
  predictVal.value = -1;
});

const handleToTrain = async () => {
  isTrain.value = true;
  emits("update", { loading: isTrain.value });
  try {
    const { data } = await cityTrain();
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
    const { data } = await cityPredict(city.value);
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
        text: "City Val Losses and City All Losses",
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

const fetchCities = async () => {
  const { data } = await cityList();
  cities.value = data;
};

onBeforeMount(() => {
  fetchCities();
});
</script>

<template>
  <content-card>
    <h2 class="select-none font-weight-medium">What is this model?</h2>
    <p class="text-indigo mt-2">
      <strong>Country classifier</strong>: Classify city names to country
    </p>
    <p class="text-indigo mt-2">
      This dataset has a list of city names and their countries as label.
      <br />
      The following countries are included in the dataset.
    </p>
    <table class="mt-2 border w-50" style="border-spacing: 0">
      <thead class="select-none">
        <tr class="bg-deep-purple-accent-1">
          <th class="text-start px-2 py-1">ID</th>
          <th class="text-start px-2 py-1">Country</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="country in countries" :key="country.id">
          <td class="px-2 py-1">{{ country.id }}</td>
          <td class="px-2 py-1">{{ country.name }}</td>
        </tr>
      </tbody>
    </table>

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

    <v-select
      class="mt-3"
      variant="outlined"
      density="comfortable"
      label="City"
      :items="cities"
      v-model="city"
      :disabled="isTrain"
      placeholder="Select a city"
    ></v-select>

    <v-btn
      :loading="isTrain"
      variant="tonal"
      color="purple"
      @click="handlePredict"
      :disabled="city === null || city === ''"
    >
      Predict
    </v-btn>

    <p class="mt-2" v-if="predictVal !== -1">
      Country: <strong>{{ predictVal }}</strong>
    </p>
  </content-card>
</template>

<style scoped>
tr:nth-child(even) {
  background-color: #f2f2f2;
}

tr:nth-child(odd) {
  background-color: #ffffff;
}
</style>
