<script setup>
import SimpleAndHardWord from "@/pages/classify/components/SimpleAndHardWord.vue";

const text = ref("1");
</script>

<template>
  <v-btn-toggle class="mb-8" v-model="text" color="#8381C5" rounded="0" group>
    <v-btn value="1">Simple & Hard Word</v-btn>
    <v-btn value="2">City names 2 country</v-btn>
  </v-btn-toggle>

  <simple-and-hard-word v-if="text === '1'" />
</template>

<style scoped></style>
