<template>
  <v-app-bar color="#8381C5" density="comfortable">
    <v-container>
      <v-responsive class="mx-auto" max-width="900">
        <v-btn @click="$router.push('/')">
          <v-icon icon="mdi-arrow-left" start></v-icon>
          Back
        </v-btn>
      </v-responsive>
    </v-container>
  </v-app-bar>

  <v-main>
    <v-container>
      <v-responsive class="mx-auto py-3" max-width="900">
        <router-view />
      </v-responsive>
    </v-container>
  </v-main>
</template>
<script setup lang="ts"></script>
