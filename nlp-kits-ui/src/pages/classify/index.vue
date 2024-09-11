<script setup>
import SimpleAndHardWord from "@/pages/classify/components/SimpleAndHardWord.vue";
import City from "@/pages/classify/components/City.vue";

const text = ref("1");
</script>

<template>
  <v-btn-toggle class="mb-8" v-model="text" color="#8381C5" rounded="0" group>
    <v-btn value="1">Simple & Hard Word</v-btn>
    <v-btn value="2">City names 2 country</v-btn>
  </v-btn-toggle>

  <simple-and-hard-word v-show="text === '1'" />
  <city v-show="text === '2'" />
</template>

<style scoped></style>
